// Micrograd in Zig

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const AutoHashMapUnmanaged = std.AutoHashMapUnmanaged;

const Error = Allocator.Error;

const Op = enum {
    Leaf,
    Add,
    Mul,
};

const ValRef = struct {
    index: u64,
};

const ValRefSet = struct {
    const Self = @This();
    const Set = AutoHashMapUnmanaged(ValRef, void);

    set: Set,

    const empty = Self{ .set = Set{} };

    fn from_slice(allocator: Allocator, comptime L: comptime_int, slice: [L]ValRef) Error!Self {
        var set = Self.empty;

        inline for (slice) |ref| {
            try set.set.put(allocator, ref, {});
        }

        return set;
    }

    fn size(self: Self) Set.Size {
        return self.set.size;
    }

    fn contains(self: Self, ref: ValRef) bool {
        return self.set.contains(ref);
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        self.set.deinit(allocator);
    }
};

const ValueBuf = struct {
    const init_capacity = 64;

    allocator: Allocator,

    data: ArrayListUnmanaged(f64),
    grads: ArrayListUnmanaged(f64),
    prev: ArrayListUnmanaged(ValRefSet),
    ops: ArrayListUnmanaged(Op),

    const Self = @This();

    fn init(allocator: Allocator) Error!Self {
        const data = try ArrayListUnmanaged(f64).initCapacity(allocator, init_capacity);
        const grads = try ArrayListUnmanaged(f64).initCapacity(allocator, init_capacity);
        const prevs = try ArrayListUnmanaged(ValRefSet).initCapacity(allocator, init_capacity);
        const ops = try ArrayListUnmanaged(Op).initCapacity(allocator, init_capacity);

        return Self{
            .allocator = allocator,
            .data = data,
            .grads = grads,
            .prev = prevs,
            .ops = ops,
        };
    }

    fn deinit(self: *Self) void {
        self.data.deinit(self.allocator);
        self.grads.deinit(self.allocator);
        for (self.prev.items) |*set| {
            set.deinit(self.allocator);
        }
        self.prev.deinit(self.allocator);
        self.ops.deinit(self.allocator);
    }

    fn value(self: *Self, data: f64, children: ValRefSet, op: Op) Error!ValRef {
        const index = self.data.items.len;

        try self.data.append(self.allocator, data);
        try self.grads.append(self.allocator, 0);
        try self.prev.append(self.allocator, children);
        try self.ops.append(self.allocator, op);

        return ValRef{ .index = index };
    }

    fn get_data(self: *Self, ref: ValRef) f64 {
        return self.data.items[ref.index];
    }

    fn get_grad(self: *Self, ref: ValRef) f64 {
        return self.grads.items[ref.index];
    }

    fn get_prev(self: *Self, ref: ValRef) *ValRefSet {
        return &self.prev.items[ref.index];
    }

    fn get_op(self: *Self, ref: ValRef) Op {
        return self.ops.items[ref.index];
    }

    fn leaf(self: *Self, data: f64) Error!ValRef {
        return self.value(data, ValRefSet.empty, Op.Leaf);
    }

    fn add(self: *Self, a: ValRef, b: ValRef) Error!ValRef {
        const children = try ValRefSet.from_slice(self.allocator, 2, .{ a, b });

        return self.value(
            self.get_data(a) + self.get_data(b),
            children,
            Op.Add,
        );
    }

    fn mul(self: *Self, a: ValRef, b: ValRef) Error!ValRef {
        const children = try ValRefSet.from_slice(self.allocator, 2, .{ a, b });

        return self.value(
            self.get_data(a) * self.get_data(b),
            children,
            Op.Mul,
        );
    }
};

test "add" {
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const a = try buf.leaf(2);
    try testing.expectEqual(Op.Leaf, buf.get_op(a));

    const b = try buf.leaf(-3);
    try testing.expectEqual(Op.Leaf, buf.get_op(b));

    const added = try buf.add(a, b);
    try testing.expectEqual(Op.Add, buf.get_op(added));

    const data = buf.get_data(added);
    try testing.expectEqual(-1, data);

    const prev = buf.get_prev(added);
    try testing.expectEqual(2, prev.size());
    try testing.expect(prev.contains(a));
    try testing.expect(prev.contains(b));
}

test "mul" {
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const a = try buf.leaf(2);
    try testing.expectEqual(Op.Leaf, buf.get_op(a));

    const b = try buf.leaf(-3);
    try testing.expectEqual(Op.Leaf, buf.get_op(b));

    const muled = try buf.mul(a, b);
    try testing.expectEqual(Op.Mul, buf.get_op(muled));

    const data = buf.get_data(muled);
    try testing.expectEqual(-6, data);

    const prev = buf.get_prev(muled);
    try testing.expectEqual(2, prev.size());
    try testing.expect(prev.contains(a));
    try testing.expect(prev.contains(b));
}

test "mul add" {
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const a = try buf.leaf(2);
    try testing.expectEqual(Op.Leaf, buf.get_op(a));

    const b = try buf.leaf(-3);
    try testing.expectEqual(Op.Leaf, buf.get_op(b));

    const c = try buf.leaf(10);
    try testing.expectEqual(Op.Leaf, buf.get_op(c));

    const muled = try buf.mul(a, b);
    try testing.expectEqual(Op.Mul, buf.get_op(muled));

    const added = try buf.add(muled, c);
    try testing.expectEqual(Op.Add, buf.get_op(added));

    const val = buf.get_data(added);
    try testing.expectEqual(4, val);

    const prev = buf.get_prev(added);
    try testing.expectEqual(2, prev.size());
    try testing.expect(prev.contains(muled));
    try testing.expect(prev.contains(c));
}

test "mul add mul" {
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const a = try buf.leaf(2);
    const b = try buf.leaf(-3);
    const c = try buf.leaf(10);
    const muled = try buf.mul(a, b);
    const added = try buf.add(muled, c);
    const f = try buf.leaf(-2);
    const L = try buf.mul(added, f);

    const val = buf.get_data(L);
    try testing.expectEqual(-8, val);
}
