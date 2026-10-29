const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
fn pick(history: []const u32, s: lanes.Sampling, penalties: lanes.Penalties) !u32 {
 var values: [7]f64 = undefined;
 var ids: [7]u64 = undefined;
 var hash: usize = 0;
 for (history) |t| hash = (hash * 11 + t) % 31;
 for (&values, &ids, 0..) |*v,*id,i| {
  var count: u32 = 0;
  for (history[2..]) |t| { if(t==i) count += 1; }
  v.* = @as(f64,@floatFromInt((hash+i*5)%17))/4 - penalties.of(count);
  id.*=i;
 }
 return @intCast(try lanes.sampling.choose(a,&values,&ids,history.len,s));
}
test "review keyed target draws agree across deterministic speculative blocks" {
 var cases: usize = 0;
 for (0..64) |seed| for ([_]f64{0,0.2,0.7,1.5}) |temp| for ([_]f64{0.7,1}) |p| for ([_]u32{0,3}) |k| for ([_]bool{false,true}) |pen| {
  const s: lanes.Sampling = .{.seed=seed,.temperature=temp,.top_p=p,.top_k=k};
  const penalties: lanes.Penalties = if(pen) .{.presence=0.5,.frequency=0.25} else .{};
  var serial: [34]u32 = undefined; serial[0]=1;serial[1]=2;
  for (2..34) |i| serial[i]=try pick(serial[0..i],s,penalties);
  var drafted: [42]u32 = undefined;drafted[0]=1;drafted[1]=2;drafted[2]=serial[2];
  var len: usize=3;
  while (len<34) {
   var tokens: [8]u32=undefined;var picks: [8]u32=undefined;
   const parents=[_]i32{-1,0,1,2,3,4,5,6};
   tokens[0]=drafted[len-1];
   var assumed=drafted;
   for (0..8) |i| {
    if(i>0) { tokens[i]=@intCast((len+i*3)%7);assumed[len+i-1]=tokens[i]; }
    picks[i]=try pick(assumed[0..len+i],s,penalties);
   }
   const path=try lanes.accept.acceptPath(a,&tokens,&parents,&picks);defer a.free(path);
   for (path) |row| { if(len==34) break;drafted[len]=picks[row];len+=1; }
  }
  try std.testing.expectEqualSlices(u32,&serial,drafted[0..34]);cases+=1;
 };
 std.debug.print("keyed equality cases={d}, generated tokens={d}\n",.{cases,cases*32});
}
const Stop = struct { fn check(_: *anyopaque, tokens: []const u32) bool { return std.mem.endsWith(u32,tokens,&.{4,5}); } };
test "review accepted blocks stop at each EOS and token budget and across a stop boundary" {
 const block=[_]u32{2,3,4,5,6,7,8,9};
 for (1..9) |cut| {
  var limited=try lanes.Stream.init(a,.{.id="limit",.prompt=&.{1},.max_new=@intCast(cut),.eos=&.{}});defer limited.deinit(a);
  try std.testing.expectEqual(cut,try limited.commit(a,&block,&.{}));
  try std.testing.expectEqual(.length,limited.reason);
  var eos=try lanes.Stream.init(a,.{.id="eos",.prompt=&.{1},.max_new=16,.eos=&.{block[cut-1]}});defer eos.deinit(a);
  try std.testing.expectEqual(cut,try eos.commit(a,&block,&.{}));
  try std.testing.expectEqual(.stop,eos.reason);
 }
 var marker:u8=0;
 for (0..5) |split| {
  var stopped=try lanes.Stream.init(a,.{.id="stop",.prompt=&.{1},.max_new=16,.eos=&.{},.stop_check=.{.ptr=&marker,.check=Stop.check}});defer stopped.deinit(a);
  _=try stopped.commit(a,block[0..split],&.{});_=try stopped.commit(a,block[split..],&.{});
  try std.testing.expectEqualSlices(u32,block[0..4],stopped.emitted());try std.testing.expectEqual(.stop,stopped.reason);
 }
 std.debug.print("termination cases=21\n",.{});
}
