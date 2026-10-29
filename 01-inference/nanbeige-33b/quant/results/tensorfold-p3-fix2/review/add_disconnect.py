from pathlib import Path
r=Path('/tmp/nanbeige-review-97d4ac9c/head/zig/src/server')
p=r/'llama_output_test.zig';s=p.read_text().replace('events: std.ArrayList(Value) = .empty,','events: std.ArrayList(Value) = .empty,\n    close_after: ?usize = null,')
s=s.replace('if (value) |v| {\n            const bytes', 'if (g.close_after) |n| if (g.events.items.len >= n) return error.Closed;\n        if (value) |v| {\n            const bytes')
s+='''
pub fn runClosed(srv: *server.Server, a: Allocator, body: Value, after: usize) Got {
 var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
 var got: Got = .{ .a = a, .close_after = after };
 openai.run(srv, a, got.out(), .{ .conn = &conn }, true, body);
 return got;
}
''';p.write_text(s)
p=r/'llama_blocks_test.zig';s=p.read_text()+'''
test "review sink disconnect within eight token block preserves next request" {
 var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
 defer arena.deinit();
 const a = arena.allocator();
 const text = try @import("llama_prompt_test.zig").load(a);
 defer text.deinit();
 const ids = try text.text().encode(a, "a", false);
 var tokens: [24]u32 = @splat(ids[0]);
 var script: support.Script = .{ .reply = &tokens, .block = 8 };
 const srv = try @import("server.zig").Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{ .served_name = "nanbeige", .model_type = "llama", .model_ids = &.{"nanbeige"}, .use_drafts = false }, null);
 defer srv.deinit();
 const body = (try json.parse(a, "{\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"x\\"}],\\"max_tokens\\":24,\\"stream\\":true,\\"stream_options\\":{\\"include_usage\\":true}}")).ok;
 const normal = try wire(srv, a, body, true);
 for (1..9) |cut| {
  const got = support.runClosed(srv, a, body, cut);
  try std.testing.expectEqual(cut, got.events.items.len);
  try std.testing.expectEqualStrings(normal, try wire(srv, a, body, true));
 }
 std.debug.print("review disconnect offsets=8 next request bytes equal\\n", .{});
}
''';p.write_text(s)
