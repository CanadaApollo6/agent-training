const Model = @import("forward.zig").Model;
pub const Runtime = struct {
 graph: struct { config: struct { window: usize = 2048 } = .{} } = .{},
 context: struct { cache: usize = 1, begin: usize = 0, end: usize = 0 } = .{},
 pub fn reset(r: *Runtime) void { r.context = .{}; }
};
pub const Attachment = struct {
 m: *Model, r: *Runtime, held: [7]u32 = @splat(42), count: usize = 0,
 pub fn capture(_: *Attachment, _: usize) void {}
 pub fn ingest(d: *Attachment, _: usize, position: usize) !void {
  if (position != d.r.context.end) return error.BadCommittedTaps;
  d.r.context.end = position + 1;
 }
 pub fn propose(d: *Attachment, _: u32, depth: u32) !void { d.count = @min(depth, 7); }
};
