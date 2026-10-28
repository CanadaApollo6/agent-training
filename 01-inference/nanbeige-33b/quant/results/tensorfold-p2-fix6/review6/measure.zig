const std = @import("std");
const json = @import("json");
const literal = @import("source/zig/src/server/nanbeige_literal.zig");
const unicode = @import("source/zig/src/server/nanbeige_unicode.zig");
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;
fn ns() i64 {
    var t: Timespec = undefined;
    std.debug.assert(clock_gettime(1, &t) == 0);
    return t.sec * 1000000000 + t.nsec;
}
fn worker(count: usize) void {
    for (0..count) |_| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const v = (literal.parse(arena.allocator(), "'\\N{SNOWMAN}\\N{LATIN CAPITAL LETTER A}'") catch @panic("oom")).?;
        std.debug.assert(v == .string and std.mem.eql(u8, v.string, "☃A"));
    }
}
test "independent allocation and lifetime" {
    for ([_][]const u8{ "'A'", "'\\N{CJK UNIFIED IDEOGRAPH-4E00}'", "'\\N{SNOWMAN}'", "'\\N{NOT A REAL NAME}'", "'\\N{SNOWMAN}\\N{SNOWMAN}'" }) |input| {
        for (0..5) |repeat| {
            const start = ns();
            var capacity: usize = 0;
            var checks: usize = 0;
            for (0..100) |_| {
                var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
                defer arena.deinit();
                const value = try literal.parse(arena.allocator(), input);
                if (std.mem.indexOf(u8, input, "NOT A REAL") != null) {
                    try std.testing.expectEqual(null, value);
                } else try std.testing.expect(value != null and value.? == .string);
                checks += 1;
                capacity = arena.queryCapacity();
            }
            std.debug.print("MEASURE {d} {s} ns_per_parse={d} arena_capacity={d} checked={d}\n", .{repeat,input,@divTrunc(ns()-start,100),capacity,checks});
        }
    }
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var names: unicode.Names = .{};
    try std.testing.expectEqual(@as(u21, 0x2603), (try names.lookup(arena.allocator(), "SNOWMAN")).?);
    const first = arena.queryCapacity();
    const pointer = names.bytes.?.ptr;
    for (0..1000) |_| try std.testing.expectEqual(@as(u21, 0x2603), (try names.lookup(arena.allocator(), "SNOWMAN")).?);
    try std.testing.expectEqual(first, arena.queryCapacity());
    try std.testing.expect(pointer == names.bytes.?.ptr);
    std.debug.print("TABLE bytes={d} arena_capacity={d} repeated_lookups=1000 no_growth=true\n", .{names.bytes.?.len,first});
    for ([_]usize{1, 8, 32, 64}) |count| {
        var many = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer many.deinit();
        const start_many = ns();
        for (0..count) |_| {
            const v = (try literal.parse(many.allocator(), "'\\N{SNOWMAN}'")).?;
            try std.testing.expectEqualStrings("☃", v.string);
        }
        std.debug.print("MANY literals={d} arena_capacity={d} elapsed_ns={d}\n", .{count,many.queryCapacity(),ns()-start_many});
    }
    var threads: [8]std.Thread = undefined;
    const start = ns();
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, worker, .{100});
    for (threads) |t| t.join();
    std.debug.print("CONCURRENT threads=8 checks=800 elapsed_ns={d}\n", .{ns()-start});
}
