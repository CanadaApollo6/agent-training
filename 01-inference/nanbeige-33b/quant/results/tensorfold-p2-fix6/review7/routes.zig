const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const server = @import("source/zig/src/server/server.zig");
const hf = @import("source/zig/src/server/llama_prompt_test.zig");
const Conn = @import("source/zig/src/server/http_conn.zig").Conn;
const Value = json.Value;
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;
fn ns() i64 {
    var t: Timespec = undefined;
    std.debug.assert(clock_gettime(1, &t) == 0);
    return t.sec * 1000000000 + t.nsec;
}
const Script = struct {
    reply: []const u32 = &.{},
    prompt: []const u32 = &.{},
    a: std.mem.Allocator,
    fn engine(e: *Script) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info { return .{ .context_window = 262144, .penalties = true }; }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        e.prompt = e.a.dupe(u32, req.prompt) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.reply[0..@min(e.reply.len, req.max_tokens)]) |*t| sink.event(sink.ctx, id, &.{ .tokens = t[0..1] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (e.reply.len >= req.max_tokens) .length else .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
};
const Wire = struct {
    a: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.text.appendSlice(w.a, bytes) catch return error.Closed;
    }
};
fn wire(srv: *server.Server, a: std.mem.Allocator, body: Value) ![]const u8 {
    var w: Wire = .{ .a = a };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "POST",
        .write_hooks = .{ .ctx = &w, .write = Wire.write } };
    var client: @import("source/zig/src/server/openai_client.zig").ClientOut = .{ .conn = &conn, .a = a, .pod = true };
    @import("source/zig/src/server/openai.zig").run(srv, a, client.out(), .{ .conn = &conn }, true, body);
    return w.text.items;
}
test "independent production HTTP writer and prompt capture" {
    var outer = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer outer.deinit();
    const a = outer.allocator();
    const text = try hf.load(a);
    defer text.deinit();
    var script: Script = .{ .a = a };
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige", .model_type = "llama", .model_ids = &.{"nanbeige"}, .use_drafts = false,
    }, null);
    defer srv.deinit();
    const path = std.mem.span(std.c.getenv("REVIEW_INPUT").?);
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1024 * 1024 * 1024));
    const history_data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "/tmp/nanbeige-review-9a5732e0/histories.json", a, .limited(100 * 1024 * 1024));
    const histories = (try json.parse(a, history_data)).ok;
    var seen: [55]bool = @splat(false);
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const b = arena.allocator();
        const case = (try json.parse(b, line)).ok;
        const raw_ids = case.get("ids").?.array;
        const ids = try b.alloc(u32, raw_ids.len);
        for (raw_ids, ids) |v, *t| t.* = @intCast(v.int64().?);
        script.reply = ids;
        script.prompt = &.{};
        script.a = b;
        const body = if (case.get("history")) |h| try json.deepCopy(b, histories.array[@intCast(h.int64().?)]) else case.get("request").?;
        if (case.get("request")) |overrides| for (overrides.object.keys(), overrides.object.values()) |k, v| { try body.object.put(b, k, v); };
        const start = ns();
        const whole = try wire(srv, b, body);
        try std.testing.expectEqual(@as(u32, 0), srv.preparing.load(.acquire));
        const elapsed = ns() - start;
        const whole_capacity = arena.queryCapacity();
        const captured = script.prompt;
        try body.object.put(b, "stream", .{ .bool = true });
        const stream_start = ns();
        const streamed = try wire(srv, b, body);
        try std.testing.expectEqual(@as(u32, 0), srv.preparing.load(.acquire));
        try std.testing.expect(std.unicode.utf8ValidateSlice(whole));
        try std.testing.expect(std.unicode.utf8ValidateSlice(streamed));
        const stream_elapsed = ns() - stream_start;
        std.debug.print("TIMING {s} json_ns={d} sse_ns={d} json_capacity={d} total_capacity={d}\n", .{case.strField("name").?, elapsed, stream_elapsed, whole_capacity, arena.queryCapacity()});
        const result = try json.newObject(b);
        try result.put(b, "name", case.get("name").?);
        try result.put(b, "json", .{ .string = whole });
        try result.put(b, "sse", .{ .string = streamed });
        const hidx: usize = if (case.get("history")) |h| @intCast(h.int64().?) else 0;
        const capture = case.get("history") == null or !seen[hidx];
        seen[hidx] = true;
        const prompt = try b.alloc(Value, if (capture) captured.len else 0);
        for (captured[0..prompt.len], prompt) |t, *v| v.* = try json.intValue(b, t);
        try result.put(b, "prompt", .{ .array = prompt });
        std.debug.print("RESULT {s}\n", .{try json.stringify(b, .{ .object = result }, .{ .ascii = false, .compact = true })});
    }
}
