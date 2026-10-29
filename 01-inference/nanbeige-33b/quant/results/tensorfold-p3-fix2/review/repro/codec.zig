const std = @import("std");
const kv = @import("kv");
const a = std.testing.allocator;
test "codec row and group isolation with rollback at every tail length" {
 var cases: usize = 0;
 inline for (.{128,256}) |dim| {
  const bytes = dim/2+dim/16;
  var x: [dim]f32 = undefined;
  var decoded: [dim]f32 = undefined;
  var cache: [bytes*17+64]u8 = @splat(0xa5);
  for (0..17) |row| {
   for (&x,0..) |*v,i| v.* = @as(f32,@floatFromInt((row*13+i*7)%97))-48;
   kv.pack(&x,cache[32+row*bytes..][0..bytes]);
   kv.unpack(cache[32+row*bytes..][0..bytes],&decoded);
   for (decoded) |v| try std.testing.expect(std.math.isFinite(v));
  }
  for (0..17) |keep| {
   const original = cache;
   for (&x,0..) |*v,i| v.* = @as(f32,@floatFromInt(i%7));
   for (keep..17) |row| kv.pack(&x,cache[32+row*bytes..][0..bytes]);
   try std.testing.expectEqualSlices(u8,original[0..32+keep*bytes],cache[0..32+keep*bytes]);
   try std.testing.expectEqualSlices(u8,original[32+17*bytes..],cache[32+17*bytes..]);
   cache = original;
   cases += 1;
  }
  for (0..dim/32) |group| {
   for ([_]f32{std.math.nan(f32),std.math.inf(f32),-std.math.inf(f32),1000000}) |bad| {
    @memset(&x,0); x[group*32+7]=bad;
    var encoded: [bytes]u8 = undefined;
    kv.pack(&x,&encoded); kv.unpack(&encoded,&decoded);
    for (decoded,0..) |v,i| {
     if (i/32==group) try std.testing.expect(std.math.isNan(v)) else try std.testing.expectEqual(@as(f32,0),v);
    }
    cases += 1;
   }
  }
 }
 std.debug.print("codec boundary/nonfinite cases={d}\n",.{cases});
}
test "rotation is its inverse for every basis and 1000 deterministic vectors" {
 for (0..1032) |j| {
  var x: [32]f32 = undefined;
  for (&x,0..) |*v,i| v.* = if(j<32) @floatFromInt(@intFromBool(i==j)) else @as(f32,@floatFromInt((j*13+i*7)%97))-48;
  const original=x; kv.turn(&x); kv.turn(&x);
  for (x,original) |v,expected| try std.testing.expectApproxEqAbs(expected,v,0.00002);
 }
}
