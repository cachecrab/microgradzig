// Micrograd in Zig

// export fn godbolt() f64 {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     const allocator = gpa.allocator();

//     var buf = ValueBuf.init(allocator) catch unreachable;
//     defer buf.deinit();

//     const x1 = buf.leaf(2) catch unreachable;
//     const x2 = buf.leaf(0) catch unreachable;
//     const w1 = buf.leaf(-3) catch unreachable;
//     const w2 = buf.leaf(1) catch unreachable;
//     const b = buf.leaf(6.8813735870195432) catch unreachable;
//     const x1w1 = buf.mul(x1, w1) catch unreachable;
//     const x2w2 = buf.mul(x2, w2) catch unreachable;
//     const x1w1x2w2 = buf.add(x1w1, x2w2) catch unreachable;
//     const n = buf.add(x1w1x2w2, b) catch unreachable;
//     const o = buf.tanh(n) catch unreachable;

//     var order = buf.build_topological_order(allocator, o) catch unreachable;
//     defer order.deinit(allocator);

//     buf.backward(order.items);

//     return buf.get_grad(x1).*;
// }

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

    inner: Set,

    const empty = Self{ .inner = Set{} };

    fn from_slice(allocator: Allocator, slice: anytype) Error!Self {
        var set = Self.empty;

        inline for (slice) |ref| {
            try set.inner.put(allocator, ref, {});
        }

        return set;
    }

    fn size(self: Self) Set.Size {
        return self.inner.size;
    }

    fn put(self: *Self, allocator: Allocator, ref: ValRef) Error!void {
        try self.inner.put(allocator, ref, {});
    }

    fn contains(self: Self, ref: ValRef) bool {
        return self.inner.contains(ref);
    }

    fn iterator(self: Self) Set.KeyIterator {
        return self.inner.keyIterator();
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        self.inner.deinit(allocator);
    }
};

const ChildrenRef = packed struct {
    unary_or_binary: u1,
    index: u63,

    const Self = @This();

    const empty = Self{ .unary_or_binary = 0, .index = 0 };

    fn unary(index: u63) Self {
        return Self{
            .unary_or_binary = 0,
            .index = index + 1,
        };
    }

    fn binary(index: u63) Self {
        return Self{
            .unary_or_binary = 1,
            .index = index + 1,
        };
    }

    fn get_index(self: Self) u63 {
        return self.index - 1;
    }

    fn is_empty(self: Self) bool {
        return self.unary_or_binary == 0 and self.index == 0;
    }

    fn is_unary(self: Self) bool {
        return self.unary_or_binary == 0 and self.index != 0;
    }

    fn is_binary(self: Self) bool {
        return self.unary_or_binary == 1;
    }
};

const ChildrenBuf = struct {
    const init_capacity = 32;

    unary: ArrayListUnmanaged(ValRef),
    binary: ArrayListUnmanaged([2]ValRef),

    const Self = @This();

    fn init(allocator: Allocator) Error!Self {
        const unary = try ArrayListUnmanaged(ValRef).initCapacity(allocator, init_capacity);
        const binary = try ArrayListUnmanaged([2]ValRef).initCapacity(allocator, init_capacity);

        return Self{
            .unary = unary,
            .binary = binary,
        };
    }

    fn new_unary(self: *Self, allocator: Allocator, ref: ValRef) Error!ChildrenRef {
        const index: u63 = @intCast(self.unary.items.len);
        try self.unary.append(allocator, ref);

        return ChildrenRef.unary(index);
    }

    fn new_binary(self: *Self, allocator: Allocator, refs: [2]ValRef) Error!ChildrenRef {
        const index: u63 = @intCast(self.binary.items.len);
        try self.binary.append(allocator, refs);

        return ChildrenRef.binary(index);
    }

    fn get(self: Self, ref: ChildrenRef) []ValRef {
        if (ref.is_empty()) {
            return &[0]ValRef{};
        }

        const index = ref.get_index();

        switch (ref.unary_or_binary) {
            0 => return self.unary.items[index .. index + 1],
            1 => return &self.binary.items[index],
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
        const children = try self.children_buf.new_binary(self.allocator, .{ a, b });

        return self.value(
            self.get_data(a) + self.get_data(b),
            children,
            Op.Add,
        );
    }

    fn mul(self: *Self, a: ValRef, b: ValRef) Error!ValRef {
        const children = try self.children_buf.new_binary(self.allocator, .{ a, b });

        return self.value(
            self.get_data(a) * self.get_data(b),
            children,
            Op.Mul,
        );
    }

    fn tanh(self: *Self, val: ValRef) Error!ValRef {
        const children = try self.children_buf.new_unary(self.allocator, val);

        const data = self.get_data(val);
        const e_2data = @exp(2 * data);
        const out = (e_2data - 1) / (e_2data + 1);

        return self.value(out, children, Op.Tanh);
    }

    fn propagate(self: *Self, val: ValRef) void {
        const grad = self.get_grad(val).*;
        const prev = self.get_prev(val);
        const children = self.children_buf.get(prev);

        switch (self.get_op(val)) {
            Op.Leaf => {},
            Op.Add => {
                const a, const b = children[0..2].*;
                self.get_grad(a).* += grad;
                self.get_grad(b).* += grad;
            },
            Op.Mul => {
                const a, const b = children[0..2].*;
                self.get_grad(a).* += self.get_data(b) * grad;
                self.get_grad(b).* += self.get_data(a) * grad;
            },
            Op.Tanh => {
                const data = self.get_data(val);
                self.get_grad(children[0]).* += (1 - data * data) * grad;
            },
        }
    }

    /// Returns ownership, call `result.deinit(allocator);`
    fn build_rev_topo_order(self: *Self, allocator: Allocator, start: ValRef) Error!ArrayListUnmanaged(ValRef) {
        var topo = try ArrayListUnmanaged(ValRef).initCapacity(allocator, self.data.items.len);

        var visited = ValRefSet.empty;
        defer visited.deinit(allocator);

        var fifo = std.fifo.LinearFifo(ValRef, std.fifo.LinearFifoBufferType.Dynamic).init(allocator);
        defer fifo.deinit();

        try fifo.writeItem(start);
        while (fifo.readItem()) |val| {
            try visited.put(allocator, val);

            const children = self.children_buf.get(self.get_prev(val));
            for (children) |child| {
                if (!visited.contains(child)) {
                    try fifo.writeItem(child);
                }
            }
            try topo.append(allocator, val);
        }

        return topo;
    }

    fn backward(self: *Self, rev_topo_order: []const ValRef) void {
        self.get_grad(rev_topo_order[0]).* = 1;

        for (rev_topo_order) |ref| {
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
    const children = buf.children_buf.get(prev);
    try testing.expectEqual(a, children[0]);
    try testing.expectEqual(b, children[1]);
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
    const children = buf.children_buf.get(prev);
    try testing.expectEqual(a, children[0]);
    try testing.expectEqual(b, children[1]);
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
    const children = buf.children_buf.get(prev);
    try testing.expectEqual(muled, children[0]);
    try testing.expectEqual(c, children[1]);
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
    const children = buf.children_buf.get(prev);
    try testing.expectEqual(n, children[0]);

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
    try testing.expectApproxEqAbs(-1.5, buf.get_grad(x1).*, 2 * eps);
    try testing.expectApproxEqAbs(1.0, buf.get_grad(w1).*, eps);

    buf.propagate(x2w2);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x2).*, eps);
    try testing.expectApproxEqAbs(0, buf.get_grad(w2).*, eps);
}

test "build reversed topological order" {
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

    var order = try buf.build_rev_topo_order(testing.allocator, o);
    defer order.deinit(testing.allocator);

    // Check the total number of nodes
    try testing.expectEqual(10, order.items.len);

    // Check that the order is valid (parents come before children)
    var visited = std.AutoHashMap(ValRef, void).init(testing.allocator);
    defer visited.deinit();

    var i = order.items.len - 1;
    while (i != 0) : (i -= 1) {
        const node = order.items[i];
        const prev = buf.get_prev(node);
        const children = buf.children_buf.get(prev);
        for (children) |child| {
            try testing.expect(visited.contains(child));
        }
        try visited.put(node, {});
    }
}

test "backwards" {
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

    var order = try buf.build_rev_topo_order(testing.allocator, o);
    defer order.deinit(testing.allocator);

    buf.backward(order.items);

    try testing.expectApproxEqAbs(-1.5, buf.get_grad(x1).*, 0.01);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x2).*, 0.01);
    try testing.expectApproxEqAbs(1, buf.get_grad(w1).*, 0.01);
    try testing.expectApproxEqAbs(0, buf.get_grad(w2).*, 0.01);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(b).*, 0.01);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x1w1).*, 0.01);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x2w2).*, 0.01);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(x1w1x2w2).*, 0.01);
    try testing.expectApproxEqAbs(0.5, buf.get_grad(n).*, 0.01);
    try testing.expectApproxEqAbs(1, buf.get_grad(o).*, 0.01);
}

test "non-tree backwards" {
    var buf = try ValueBuf.init(testing.allocator);
    defer buf.deinit();

    const a = try buf.leaf(3);
    const b = try buf.add(a, a);

    var order = try buf.build_rev_topo_order(testing.allocator, b);
    defer order.deinit(testing.allocator);

    buf.backward(order.items);

    try testing.expectApproxEqAbs(2, buf.get_grad(a).*, 0.01);
    try testing.expectApproxEqAbs(1, buf.get_grad(b).*, 0.01);
}
