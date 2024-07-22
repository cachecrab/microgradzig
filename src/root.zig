// Micrograd in Zig

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const AutoHashMapUnmanaged = std.AutoHashMapUnmanaged;

const assert = std.debug.assert;

const Error = Allocator.Error;

const Op = enum {
    Leaf,
    Add,
    Mul,
    Tanh,
};

const ValRef = struct {
    index: u64,
};

const ValRefSet = struct {
    const Self = @This();
    const Set = AutoHashMapUnmanaged(ValRef, void);

    set: Set,

    const empty = Self{ .set = Set{} };

    fn from_slice(allocator: Allocator, slice: anytype) Error!Self {
        var set = Self.empty;

        inline for (slice) |ref| {
            try set.set.put(allocator, ref, {});
        }

        return set;
    }

    fn size(self: Self) Set.Size {
        return self.set.size;
    }

    fn put(self: *Self, allocator: Allocator, ref: ValRef) Error!void {
        try self.set.put(allocator, ref, {});
    }

    fn contains(self: Self, ref: ValRef) bool {
        return self.set.contains(ref);
    }

    fn iterator(self: Self) Set.KeyIterator {
        return self.set.keyIterator();
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        self.set.deinit(allocator);
    }
};

const ChildrenRef = struct {
    index: u64,

    const Self = @This();

    const empty = Self{ .index = 0 };
};

const ChildrenBuf = struct {
    const init_capacity = 32;

    unary: ArrayListUnmanaged(ValRef),
    binary: ArrayListUnmanaged([2]ValRef),

    const Self = @This();

    const empty = ChildrenRef{ .index = 0 };

    fn init(allocator: Allocator) Error!Self {
        const unary = try ArrayListUnmanaged(ValRef).initCapacity(allocator, init_capacity);
        const binary = try ArrayListUnmanaged([2]ValRef).initCapacity(allocator, init_capacity);

        return Self{
            .unary = unary,
            .binary = binary,
        };
    }

    fn new_unary(self: *Self, allocator: Allocator, ref: ValRef) Error!ChildrenRef {
        const index = self.unary.items.len;
        try self.unary.append(allocator, ref);

        return ChildrenRef{ .index = ((index + 1) << 1) };
    }

    fn new_binary(self: *Self, allocator: Allocator, refs: [2]ValRef) Error!ChildrenRef {
        const index = self.binary.items.len;
        try self.binary.append(allocator, refs);

        return ChildrenRef{ .index = ((index + 1) << 1) | 1 };
    }

    fn get_unary(self: Self, ref: ChildrenRef) ValRef {
        assert(ref.index != 0);
        assert(ref.index & 1 == 0);
        return self.unary.items[(ref.index >> 1) - 1];
    }

    fn get_binary(self: Self, ref: ChildrenRef) [2]ValRef {
        assert(ref.index != 0);
        assert(ref.index & 1 == 1);
        return self.binary.items[(ref.index >> 1) - 1];
    }

    fn get(self: Self, ref: ChildrenRef) [2]?ValRef {
        if (ref.index == 0) {
            return .{ null, null };
        }

        const parity = ref.index & 1;
        switch (parity) {
            0 => return .{ self.get_unary(ref), null },
            1 => {
                const a, const b = self.get_binary(ref);
                return .{ a, b };
            },
            else => unreachable,
        }
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        self.unary.deinit(allocator);
        self.binary.deinit(allocator);
    }
};

const ValueBuf = struct {
    const init_capacity = 64;

    allocator: Allocator,

    children_buf: ChildrenBuf,

    data: ArrayListUnmanaged(f64),
    grads: ArrayListUnmanaged(f64),
    prev: ArrayListUnmanaged(ChildrenRef),
    ops: ArrayListUnmanaged(Op),

    const Self = @This();

    fn init(allocator: Allocator) Error!Self {
        const children_buf = try ChildrenBuf.init(allocator);

        const data = try ArrayListUnmanaged(f64).initCapacity(allocator, init_capacity);
        const grads = try ArrayListUnmanaged(f64).initCapacity(allocator, init_capacity);
        const prevs = try ArrayListUnmanaged(ChildrenRef).initCapacity(allocator, init_capacity);
        const ops = try ArrayListUnmanaged(Op).initCapacity(allocator, init_capacity);

        return Self{
            .allocator = allocator,
            .children_buf = children_buf,

            .data = data,
            .grads = grads,
            .prev = prevs,
            .ops = ops,
        };
    }

    fn deinit(self: *Self) void {
        self.children_buf.deinit(self.allocator);

        self.data.deinit(self.allocator);
        self.grads.deinit(self.allocator);
        self.prev.deinit(self.allocator);
        self.ops.deinit(self.allocator);
    }

    fn value(self: *Self, data: f64, children: ChildrenRef, op: Op) Error!ValRef {
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

    fn get_grad(self: *Self, ref: ValRef) *f64 {
        return &self.grads.items[ref.index];
    }

    fn get_prev(self: *Self, ref: ValRef) ChildrenRef {
        return self.prev.items[ref.index];
    }

    fn get_op(self: *Self, ref: ValRef) Op {
        return self.ops.items[ref.index];
    }

    fn leaf(self: *Self, data: f64) Error!ValRef {
        return self.value(data, ChildrenRef.empty, Op.Leaf);
    }

    fn add(self: *Self, a: ValRef, b: ValRef) Error!ValRef {
        // const children = try ValRefSet.from_slice(self.allocator, .{ a, b });
        const children = try self.children_buf.new_binary(self.allocator, .{ a, b });
        std.debug.print("add: {}\n", .{children});

        return self.value(
            self.get_data(a) + self.get_data(b),
            children,
            Op.Add,
        );
    }

    fn mul(self: *Self, a: ValRef, b: ValRef) Error!ValRef {
        // const children = try ValRefSet.from_slice(self.allocator, .{ a, b });
        const children = try self.children_buf.new_binary(self.allocator, .{ a, b });
        std.debug.print("mul: {}\n", .{children});

        return self.value(
            self.get_data(a) * self.get_data(b),
            children,
            Op.Mul,
        );
    }

    fn tanh(self: *Self, val: ValRef) Error!ValRef {
        // const children = try ValRefSet.from_slice(self.allocator, .{val});
        const children = try self.children_buf.new_unary(self.allocator, val);
        std.debug.print("tanh: {}\n", .{children});

        const data = self.get_data(val);
        const out = (@exp(2 * data) - 1) / (@exp(2 * data) + 1);

        return self.value(out, children, Op.Tanh);
    }

    fn propagate(self: *Self, val: ValRef) void {
        const grad = self.get_grad(val).*;
        const prev = self.get_prev(val);
        const a_maybe, const b_maybe = self.children_buf.get(prev);

        switch (self.get_op(val)) {
            Op.Leaf => {},
            Op.Add => {
                // const a = prev.next().?.*;
                // const b = prev.next().?.*;
                self.get_grad(a_maybe.?).* += grad;
                self.get_grad(b_maybe.?).* += grad;
            },
            Op.Mul => {
                const a = a_maybe.?;
                const b = b_maybe.?;
                self.get_grad(a).* += self.get_data(b) * grad;
                self.get_grad(b).* += self.get_data(a) * grad;
            },
            Op.Tanh => {
                // const a = prev.next().?.*;
                const data = self.get_data(val);
                self.get_grad(a_maybe.?).* += (1 - data * data) * grad;
            },
        }
    }

    /// Make sure visited doesn't contain val
    fn build_topo(self: *Self, allocator: Allocator, val: ValRef, visited: *ValRefSet, topo: *ArrayListUnmanaged(ValRef)) Error!void {
        // if (!visited.contains(val)) {
        try visited.put(allocator, val);
        const prev = self.get_prev(val);
        const children = self.children_buf.get(prev);
        // std.debug.print("-- build topo children: {any}\n", .{children});
        for (children) |child_maybe| {
            const child = child_maybe orelse break;
            if (!visited.contains(child)) {
                try self.build_topo(allocator, child, visited, topo);
            }
        }
        try topo.append(allocator, val);
        // }
    }

    /// Remember to free result by calling .deinit()
    fn build_topological_order(self: *Self, allocator: Allocator, val: ValRef) Error!ArrayListUnmanaged(ValRef) {
        var visited = ValRefSet.empty;
        defer visited.deinit(allocator);

        var topo = try ArrayListUnmanaged(ValRef).initCapacity(allocator, self.data.items.len);

        try self.build_topo(allocator, val, &visited, &topo);

        return topo;
    }

    fn backward(self: *Self, val: ValRef, topological_order: []const ValRef) void {
        self.get_grad(val).* = 1;

        var i = topological_order.len - 1;
        while (i > 0) : (i -= 1) {
            const ref = topological_order[i];
            self.propagate(ref);
        }
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
    const prev1, const prev2 = buf.children_buf.get(prev);
    try testing.expectEqual(a, prev1);
    try testing.expectEqual(b, prev2);
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

    // const prev = buf.get_prev_binary(muled);
    const prev = buf.get_prev(muled);
    const prev1, const prev2 = buf.children_buf.get(prev);
    try testing.expectEqual(a, prev1);
    try testing.expectEqual(b, prev2);
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
    const prev1, const prev2 = buf.children_buf.get(prev);
    try testing.expectEqual(muled, prev1);
    try testing.expectEqual(c, prev2);
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

test "propagate" {
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const x1 = try buf.leaf(2);
    const x2 = try buf.leaf(0);
    const w1 = try buf.leaf(-3);
    const w2 = try buf.leaf(1);
    const b = try buf.leaf(6.8813735870195432);
    const x1w1 = try buf.mul(x1, w1);
    const x2w2 = try buf.mul(x2, w2);
    const x1w1x2w2 = try buf.add(x1w1, x2w2);
    const n = try buf.add(x1w1x2w2, b);
    const o = try buf.tanh(n);

    const eps = std.math.floatEps(f64);
    try testing.expectApproxEqAbs(8.813735870195432e-1, buf.get_data(n), eps);
    try testing.expectApproxEqAbs(7.071067811865476e-1, buf.get_data(o), eps);

    const prev = buf.get_prev(o);
    const prev1, const prev2 = buf.children_buf.get(prev);
    try testing.expectEqual(n, prev1);
    try testing.expectEqual(null, prev2);

    buf.get_grad(o).* = 1;
    buf.propagate(o);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(n).*, eps);

    buf.propagate(n);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(b).*, eps);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x1w1x2w2).*, eps);

    buf.propagate(x1w1x2w2);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x1w1).*, eps);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x2w2).*, eps);

    buf.propagate(x1w1);
    try testing.expectApproxEqAbs(-1.4999999999999996e0, buf.get_grad(x1).*, eps);
    try testing.expectApproxEqAbs(1.0, buf.get_grad(w1).*, eps);

    buf.propagate(x2w2);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x2).*, eps);
    try testing.expectApproxEqAbs(0, buf.get_grad(w2).*, eps);
}

test "build topological order" {
    std.debug.print("build topological order\n\n-------------------------\n\n", .{});
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const x1 = try buf.leaf(2);
    const x2 = try buf.leaf(0);
    const w1 = try buf.leaf(-3);
    const w2 = try buf.leaf(1);
    const b = try buf.leaf(6.8813735870195432);
    const x1w1 = try buf.mul(x1, w1);
    const x2w2 = try buf.mul(x2, w2);
    const x1w1x2w2 = try buf.add(x1w1, x2w2);
    const n = try buf.add(x1w1x2w2, b);
    const o = try buf.tanh(n);

    var order = try buf.build_topological_order(testing.allocator, o);
    defer order.deinit(testing.allocator);

    std.debug.print("-------------------------", .{});

    const expected_order: [10]ValRef = .{
        b,
        x1,
        w1,
        x1w1,
        x2,
        w2,
        x2w2,
        x1w1x2w2,
        n,
        o,
    };

    // for (expected_order, order.items) |expected, actual| {
    //     try testing.expectEqual(expected, actual);
    // }
    _ = expected_order;
    std.debug.print("{any}", .{order.items});
    try testing.expect(false);
}

// test "backwards" {
//     var buf = try ValueBuf.init(testing.allocator);
//     defer buf.deinit();

//     const x1 = try buf.leaf(2);
//     const x2 = try buf.leaf(0);
//     const w1 = try buf.leaf(-3);
//     const w2 = try buf.leaf(1);
//     const b = try buf.leaf(6.8813735870195432);
//     const x1w1 = try buf.mul(x1, w1);
//     const x2w2 = try buf.mul(x2, w2);
//     const x1w1x2w2 = try buf.add(x1w1, x2w2);
//     const n = try buf.add(x1w1x2w2, b);
//     const o = try buf.tanh(n);

//     var order = try buf.build_topological_order(testing.allocator, o);
//     defer order.deinit(testing.allocator);

//     buf.backward(o, order.items);

//     const expected_grads: [10]f64 = .{
//         4.999999999999999e-1,
//         -1.4999999999999996e0,
//         9.999999999999998e-1,
//         4.999999999999999e-1,
//         4.999999999999999e-1,
//         0e0,
//         4.999999999999999e-1,
//         4.999999999999999e-1,
//         4.999999999999999e-1,
//         1e0,
//     };

//     for (expected_grads, order.items) |expected, ref| {
//         const val = buf.get_data(ref);
//         const actual = buf.get_grad(ref).*;
//         std.debug.print("\nValue: {:.4}, Grad: {:.4}\n", .{ val, actual });
//         try testing.expectApproxEqAbs(expected, actual, std.math.floatEpsAt(f64, expected));
//     }
// }

test "non-tree backwards" {
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const a = try buf.leaf(3);
    const b = try buf.add(a, a);

    var order = try buf.build_topological_order(testing.allocator, b);
    defer order.deinit(testing.allocator);

    buf.backward(b, order.items);

    try testing.expectApproxEqAbs(2, buf.get_grad(a).*, std.math.floatEpsAt(f64, 2));
    try testing.expectApproxEqAbs(1, buf.get_grad(b).*, std.math.floatEpsAt(f64, 1));
}
