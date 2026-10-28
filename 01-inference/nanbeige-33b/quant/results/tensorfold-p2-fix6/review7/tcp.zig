const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const server = @import("source/zig/src/server/server.zig");
const hf = @import("source/zig/src/server/llama_prompt_test.zig");
const listener = @import("source/zig/src/server/listener.zig");
const t = std.testing;
const Engine = struct {
    a: std.mem.Allocator,
    ids: []const u32 = &.{},
    active: std.atomic.Value(u32) = .init(0),
    cancelled: std.atomic.Value(api.Id) = .init(0),
    cancel_count: std.atomic.Value(u32) = .init(0),
    calls: std.atomic.Value(u32) = .init(0),
    slow: bool = false,
    fn engine(e: *Engine) api.Engine { return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } }; }
    fn cast(p: *anyopaque) *Engine { return @ptrCast(@alignCast(p)); }
    fn info(_: *anyopaque) api.Info { return .{ .context_window = 262144, .penalties = true }; }
    fn submit(p: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = cast(p);
        _ = e.calls.fetchAdd(1, .monotonic);
        if (e.active.fetchAdd(1, .acq_rel) != 0) { _ = e.active.fetchSub(1, .release); return error.Busy; }
        const thread = std.Thread.spawn(.{}, produce, .{ e, id, sink, @min(req.max_tokens, e.ids.len) }) catch {
            _ = e.active.fetchSub(1, .release); return error.Busy;
        };
        thread.detach();
    }
    fn produce(e: *Engine, id: api.Id, sink: api.Sink, n: usize) void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.ids[0..n]) |*token| {
            if (e.cancelled.load(.acquire) == id) {
                _ = e.active.fetchSub(1, .release);
                return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            }
            sink.event(sink.ctx, id, &.{ .tokens = token[0..1] });
            if (e.slow) std.Io.sleep(t.io, .fromMilliseconds(3), .awake) catch {};
        }
        _ = e.active.fetchSub(1, .release);
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(p: *anyopaque, id: api.Id) void { const e=cast(p); e.cancelled.store(id,.release); _ = e.cancel_count.fetchAdd(1,.monotonic); }
    fn status(p: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{ .running = cast(p).active.load(.acquire) }; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
};
fn quiesce(srv: *server.Server, e: *Engine) !void {
    var waited: usize = 0;
    while (srv.open_connections.load(.acquire)>0 or e.active.load(.acquire)>0) : (waited += 1) {
        if (waited == 10000) return error.Stuck;
        std.Io.sleep(t.io,.fromMilliseconds(1),.awake) catch {};
    }
    try t.expectEqual(@as(u32,0),srv.preparing.load(.acquire));
}
test "independent TCP server lifecycle" {
    var counted: std.heap.DebugAllocator(.{ .enable_memory_limit = true }) = .init;
    defer if (counted.deinit() == .leak) @panic("leak");
    const gpa = counted.allocator();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try hf.load(a);
    defer text.deinit();
    var e: Engine = .{ .a = gpa };
    defer gpa.free(e.ids);
    const srv = try server.Server.init(gpa,t.io,e.engine(),text.text(),.{ .served_name="nanbeige", .model_type="llama", .model_ids=&.{"nanbeige"}, .use_drafts=false, .keep_warm_s=0 },null);
    defer srv.deinit();
    const l = try listener.Listener.open(.{.ip4=.loopback(0)});
    defer l.close();
    var stop: std.atomic.Value(bool)=.init(false);
    const thread = try std.Thread.spawn(.{},server.Server.serve,.{srv,l,&stop});
    defer { stop.store(true,.release); thread.join(); }
    std.debug.print("PORT {d}\n",.{l.port()});
    var inbuf: [1024*1024]u8=undefined;
    while (true) {
        var len: usize = 0;
        while (len < inbuf.len) {
            const n=std.c.read(0,inbuf[len..].ptr,1);
            if (n<=0) return;
            if (inbuf[len]=='\n') break;
            len+=1;
        }
        try quiesce(srv,&e);
        gpa.free(e.ids);
        e.ids = &.{};
        std.debug.print("MEM {d}\n",.{counted.total_requested_bytes});
        if (std.mem.eql(u8,inbuf[0..len],"STOP")) break;
        var req_arena=std.heap.ArenaAllocator.init(t.allocator);
        defer req_arena.deinit();
        const b=req_arena.allocator();
        const config=(try json.parse(b,inbuf[0..len])).ok;
        const raw=config.strField("raw") orelse "</think>ok";
        const ids=try text.text().encode(b,raw,false);
        // Keep the script until the next quiescent control message.
        e.ids=try gpa.dupe(u32,ids);
        e.slow=if(config.get("slow")) |v| v.truthy() else false;
        std.debug.print("READY calls={d} cancels={d}\n",.{e.calls.load(.acquire),e.cancel_count.load(.acquire)});
    }
    try quiesce(srv,&e);
    std.debug.print("CLEAN calls={d} cancels={d}\n",.{e.calls.load(.acquire),e.cancel_count.load(.acquire)});
}
