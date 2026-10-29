const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
test "fully accepted prefix stays synchronized then rejects safely" {
 var m: @import("forward.zig").Model = .{ .gpa = a };
 var runtime: @import("draft.zig").Runtime = .{};
 var attachment: @import("draft.zig").Attachment = .{ .m = &m, .r = &runtime };
 var backend: @import("backend.zig").Llama = .{ .m = &m, .drafter = &attachment };
 var cfg = try lanes.Config.init(a, backend.facts(), 8, 7);
 defer cfg.deinit(a);
 var clock: lanes.fake.FixedClock = .{};
 var engine = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock());
 defer engine.deinit();
 var stream = try lanes.Stream.init(a, .{ .id = "repro", .prompt = &.{1,2}, .max_new = 32, .eos = &.{}, .drafts = true });
 defer stream.deinit(a);
 try engine.addStream(&stream);
 try engine.step();
 std.debug.print("after full acceptance target={d} draft={d} accepted={d} proposed={d}\n", .{backend.len, runtime.context.end, stream.accepted, stream.drafted});
 try std.testing.expect(stream.accepted > 0);
 try std.testing.expectEqual(stream.accepted, stream.drafted);
 try std.testing.expectEqual(runtime.context.end, backend.len);
 m.token = 43;
 try engine.step();
 try std.testing.expectEqual(runtime.context.end, backend.len);
}
