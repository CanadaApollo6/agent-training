const std = @import("std");
const lanes = @import("lanes");
pub const Model = struct {
 gpa: std.mem.Allocator, w: struct { config: struct { bos: u32 = 1, vocab: usize = 97 } = .{} } = .{},
 k: struct { draws: struct { penalize_fn: ?usize = 1 } = .{} } = .{},
 s: struct { pub fn synchronize(_: @This()) !void {} } = .{},
 context: usize = 128, tap_buffer: u64 = 0, logits: u64 = 0, token: u32 = 42,
 pub fn step(_: *Model, _: u32, _: usize, _: bool) !void {}
 pub fn batch(_: *Model, _: []const u32, _: usize) !void {}
 pub const Draw = struct { position: u32, sampling: ?lanes.Sampling, penalties: lanes.Penalties, history: []const u32, pending: ?u32 };
 pub fn pick(m: *Model, _: Draw) !u32 { return m.token; }
};
