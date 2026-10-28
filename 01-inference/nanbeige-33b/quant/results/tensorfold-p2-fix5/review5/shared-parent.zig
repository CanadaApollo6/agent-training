const std = @import("std");
const json = @import("json");
const parse = @import("parent/zig/src/server/tool_parse.zig");
test "independent shared parser bytes" {
    var outer = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer outer.deinit();
    const a = outer.allocator();
    const data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(std.c.getenv("REVIEW_INPUT").?), a, .limited(200 * 1024 * 1024));
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const b = arena.allocator();
        const c = (try json.parse(b, line)).ok;
        const result = try parse.parse(b, c.strField("raw").?, c.get("tools").?.array, null);
        const v = try json.newObject(b);
        try v.put(b,"content",.{.string=result.content});
        try v.put(b,"calls",.{.array=result.calls orelse &.{}});
        try v.put(b,"rejected",try json.intValue(b,result.rejected_envelopes));
        for (result.calls orelse &.{}) |call| try call.object.put(b,"id",.{.string="ID"});
        std.debug.print("RESULT {s}\n",.{try json.stringify(b,.{.object=v},.{.ascii=false})});
    }
}
