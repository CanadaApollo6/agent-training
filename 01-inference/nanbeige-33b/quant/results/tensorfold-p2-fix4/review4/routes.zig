const std = @import("std");
const json = @import("json");
const support = @import("source/zig/src/server/llama_output_test.zig");
const server = @import("source/zig/src/server/server.zig");
const hf = @import("source/zig/src/server/llama_prompt_test.zig");
test "independent routes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try hf.load(a);
    defer text.deinit();
    var script: support.Script = .{ .reply = &.{} };
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige", .model_type = "llama", .model_ids = &.{"nanbeige"}, .use_drafts = false,
    }, null);
    defer srv.deinit();
    const path = std.mem.span(std.c.getenv("REVIEW_INPUT").?);
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(100 * 1024 * 1024));
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var single = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer single.deinit();
        const b = single.allocator();
        const case = (try json.parse(b, line)).ok;
        const tokens = try text.text().encode(b, case.get("raw").?.string, false);
        script.reply = tokens;
        const body = (try json.parse(b, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":512}")).ok;
        try body.object.put(b, "tools", case.get("tools").?);
        if (case.get("extra")) |extra| for (extra.object.keys(), extra.object.values()) |k,v| try body.object.put(b, k, v);
        const whole = support.run(srv, b, body, true);
        try body.object.put(b, "stream", .{ .bool = true });
        const streamed = support.run(srv, b, body, true);
        const result = try json.newObject(b);
        try result.put(b, "whole", whole.body);
        try result.put(b, "events", .{ .array = streamed.events.items });
        try result.put(b, "status", try json.intValue(b, whole.status));
        try result.put(b, "ids", .{ .array = try b.alloc(json.Value, tokens.len) });
        for (tokens, result.get("ids").?.array) |t, *v| v.* = try json.intValue(b,t);
        std.debug.print("RESULT {s}\n", .{try json.stringify(b, .{ .object = result }, .{ .ascii = false, .compact = true })});
    }
}
