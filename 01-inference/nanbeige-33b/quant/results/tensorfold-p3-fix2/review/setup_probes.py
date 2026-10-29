from pathlib import Path
r=Path('/tmp/nanbeige-review-97d4ac9c')
p=r/'repro/repro.zig'
s=p.read_text().replace('prefix leaves draft context stale then rejection aborts','prefix stays synchronized then rejects safely').replace('try std.testing.expect(runtime.context.end < backend.len);','try std.testing.expectEqual(runtime.context.end, backend.len);').replace('try std.testing.expectError(error.BadCommittedTaps, engine.step());','try engine.step();\n try std.testing.expectEqual(runtime.context.end, backend.len);')
p.write_text(s)
p=r/'head/zig/src/families/llama/backend_test.zig'
s=p.read_text().replace('context: usize = 128','context: usize = 100000').replace('d.r.context.end = position + 1;','d.r.context.end = position + 1;\n        d.r.context.begin = (position + 1) -| 2047;')
s+='''
test "review 12000 mixed acceptance rounds across sliding windows and cancellation restart" {
    var m: Model = .{ .gpa = a };
    var runtime: Runtime = .{};
    var attachment: Attachment = .{ .m = &m, .r = &runtime };
    var backend: @import("backend.zig").Adapter(Model, Attachment) = .{ .m = &m, .drafter = &attachment };
    var counts: [3]usize = @splat(0);
    for (0..4) |trial| {
        var cfg = try lanes.Config.init(a, backend.facts(), 8, 7);
        defer cfg.deinit(a);
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock());
        defer engine.deinit();
        const prompt = try a.alloc(u32, 2047 + trial);
        defer a.free(prompt);
        @memset(prompt, 1);
        var stream = try lanes.Stream.init(a, .{ .id = "stress", .prompt = prompt, .max_new = 60000, .eos = &.{}, .drafts = true });
        defer stream.deinit(a);
        m.reject_at = null;
        try engine.addStream(&stream);
        for (0..3000) |round| {
            const proposals = stream.next.?.tokens.?;
            @memset(proposals, 42);
            const kind = round % 3;
            if (kind == 1 and proposals.len > 1) proposals[proposals.len / 2] = 43;
            if (kind == 2 and proposals.len > 0) proposals[0] = 43;
            const offered = stream.drafted;
            const accepted = stream.accepted;
            try engine.step();
            const delta = stream.accepted - accepted;
            const sent = stream.drafted - offered;
            if (sent > 0) {
                if (kind == 0) try std.testing.expectEqual(sent, delta);
                if (kind == 1 and proposals.len > 1) try std.testing.expect(delta > 0 and delta < sent);
                if (kind == 2) try std.testing.expectEqual(@as(u64, 0), delta);
                counts[kind] += 1;
            }
            try std.testing.expectEqual(backend.len, runtime.context.end);
            try std.testing.expectEqual(runtime.context.end -| 2047, runtime.context.begin);
        }
        for (stream.emitted()) |token| try std.testing.expectEqual(@as(u32, 42), token);
        stream.finished = true;
        stream.reason = .cancelled;
        engine.release(&stream);
        try std.testing.expect(backend.owner == null);
    }
    std.debug.print("review mixed rounds full={d} partial={d} reject={d} restarts=4 window=2048\\n", .{ counts[0], counts[1], counts[2] });
}
'''
p.write_text(s)
p=r/'head/zig/src/server/llama_output_test.zig';p.write_text(p.read_text().replace('block: usize = 1','block: usize = 8'))
p=r/'head/zig/src/server/llama_blocks_test.zig';s=p.read_text().replace('if (kind != 2) ids[cut - 1]', 'if (kind == 0) ids[cut] = xx[0];\n        if (kind != 2) ids[cut - 1]').replace('.string = "X"','.string = "XX"').replace('@intCast(cut)), whole.get("usage")','@intCast(cut + @as(usize, if (kind == 0) 1 else 0))), whole.get("usage")');p.write_text(s)
