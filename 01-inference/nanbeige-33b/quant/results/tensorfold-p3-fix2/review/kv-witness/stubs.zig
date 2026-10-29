const std = @import("std");
pub const cuda = struct {
 pub const kv4 = struct { pub const Mode = enum { bf16, pub fn parse(_: []const u8) !@This() { return .bf16; } }; };
 pub const Stream = struct { pub fn init(_: void, _: bool) !@This() { return .{}; } pub fn deinit(_: *@This()) void {} pub fn synchronize(_: @This()) !void {} };
 pub fn usage(_: bool) struct { peak: usize } { return .{ .peak = 0 }; }
};
pub const check = struct { pub const Gpu = struct { gpa: std.mem.Allocator, io: std.Io, ctx: void = {}, d: void = {} }; pub fn now(_: std.Io) i128 { return 1; } };
pub const common = struct {
 pub fn sequences(_: check.Gpu, _: []const u8) !struct { value: []const []const u32, pub fn deinit(_: @This()) void {} } { return .{ .value = &.{ &(@as([512]u32, @splat(1))) } }; }
 pub fn path(g: check.Gpu, out: []const u8, comptime fmt: []const u8, args: anytype) ![]u8 { const s = try std.fmt.allocPrint(g.gpa, fmt, args); defer g.gpa.free(s); return std.fs.path.join(g.gpa, &.{out, s}); }
};
pub const llama = struct {
 pub const weights = struct {
  pub const W = struct { config: struct { vocab: usize = 2, bos: u32 = 1 } = .{}, pub fn deinit(_: *@This()) void {} };
  pub fn loadMode(_: std.mem.Allocator, _: std.Io, _: void, _: []const u8, _: ?usize, _: usize, _: cuda.kv4.Mode) !W { return .{}; }
 };
 pub const kernels = struct { pub const Kernels = struct { pub fn load(_: void) !@This() { return .{}; } pub fn deinit(_: *@This()) void {} }; };
 pub const forward = struct { pub const Model = struct {
  logits: u64 = 0, valid: bool = false,
  pub fn initMode(_: std.mem.Allocator, _: void, _: *weights.W, _: *kernels.Kernels, _: cuda.Stream, _: usize, _: cuda.kv4.Mode) !@This() { return .{}; }
  pub fn deinit(_: *@This()) void {}
  pub fn step(m: *@This(), _: u32, _: usize, logits: bool) !void { if (logits) m.valid = true; }
  pub fn download(_: *@This(), _: u64, out: []u8) !void { @memset(out, 0); }
  pub fn greedy(m: *@This()) !u32 { if (!m.valid) return error.NoPrefillLogits; return 1; }
 }; };
};
