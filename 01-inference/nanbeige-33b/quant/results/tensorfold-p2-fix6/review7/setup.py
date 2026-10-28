from pathlib import Path
p=Path('/tmp/nanbeige-review-9a5732e0')
old=Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2-fix6/review6')
s=(old/'routes_measure.zig').read_text()
s=s.replace('const body = case.get("request").?;', '''const body = if (case.get("history")) |h| try json.deepCopy(b, histories.array[@intCast(h.int64().?)]) else case.get("request").?;
        if (case.get("request")) |overrides| for (overrides.object.keys(), overrides.object.values()) |k, v| { try body.object.put(b, k, v); };''')
s=s.replace('var lines = std.mem.splitScalar', '''const history_data = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "/tmp/nanbeige-review-9a5732e0/histories.json", a, .limited(100 * 1024 * 1024));
    const histories = (try json.parse(a, history_data)).ok;
    var seen = [_]bool{false} ** 55;
    var lines = std.mem.splitScalar''')
s=s.replace('const prompt = try b.alloc(Value, captured.len);', '''const hidx: usize = if (case.get("history")) |h| @intCast(h.int64().?) else 0;
        const capture = case.get("history") == null or !seen[hidx];
        seen[hidx] = true;
        const prompt = try b.alloc(Value, if (capture) captured.len else 0);''').replace('for (captured, prompt)', 'for (captured[0..prompt.len], prompt)')
s=s.replace('const whole = try wire(srv, b, body);','''const whole = try wire(srv, b, body);
        try std.testing.expectEqual(@as(u32, 0), srv.preparing.load(.acquire));''')
s=s.replace('const streamed = try wire(srv, b, body);','''const streamed = try wire(srv, b, body);
        try std.testing.expectEqual(@as(u32, 0), srv.preparing.load(.acquire));
        try std.testing.expect(std.unicode.utf8ValidateSlice(whole));
        try std.testing.expect(std.unicode.utf8ValidateSlice(streamed));''')
(p/'routes.zig').write_text(s)
(p/'build_routes.py').write_text((old/'build_routes.py').read_text())
