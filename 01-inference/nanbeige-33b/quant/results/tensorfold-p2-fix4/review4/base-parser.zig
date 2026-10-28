const std = @import("std");
const json = @import("json");
const pod = @import("base/zig/src/server/nanbeige.zig");
const qwen = @import("base/zig/src/server/tool_qwen.zig");
test "independent differential" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = std.mem.span(std.c.getenv("REVIEW_INPUT").?);
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(100 * 1024 * 1024));
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var single = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer single.deinit();
        const b = single.allocator();
        const case = (try json.parse(b, line)).ok;
        const split = try pod.split(b, case.get("raw").?.string);
        const parsed = try qwen.calls(b, split.answer, case.get("tools").?.array, null);
        const result = try json.newObject(b);
        try result.put(b, "reasoning", .{ .string = split.reasoning });
        try result.put(b, "content", .{ .string = parsed.content });
        try result.put(b, "calls", .{ .array = parsed.calls orelse &.{} });
        std.debug.print("RESULT {s}\n", .{try json.stringify(b, .{ .object = result }, .{ .ascii = false, .compact = true })});
    }
}
