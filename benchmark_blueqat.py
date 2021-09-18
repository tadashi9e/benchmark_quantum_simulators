import blueqat
import sys
import time

from types import TracebackType
from typing import List, Optional, Set, Type

# ----------------------------------------------------------------------
class QubitAllocator(object):
    def __init__(self) -> None:
        self.allocated: Set[int] = set()
    def reset(self) -> None:
        self.allocated = set()
    def allocate1(self) -> int:
        if not self.allocated:
            self.allocated.add(0)
            return 0
        for i in range(0, max(self.allocated)):
            if i not in self.allocated:
                self.allocated.add(i)
                return i
        i = max(self.allocated) + 1
        self.allocated.add(i)
        return i
    def deallocate1(self, index: int) -> None:
        self.allocated.remove(index)

ALLOCATOR = QubitAllocator()

def reset_all() -> None:
    ALLOCATOR.reset()
# ----------------------------------------------------------------------
class UnitaryOperation(object):
    def __init__(self) -> None:
        pass
    def reverse(self) -> None:
        pass
    def synthesis(self, bq_circuit : blueqat.Circuit) -> blueqat.Circuit:
        return bq_circuit
    def __str__(self) -> str:
        return self.string(0)
    def string(self, depth : int) -> str:
        return ''
class UO_Procedure(UnitaryOperation):
    def __init__(self, title: str) -> None:
        UnitaryOperation.__init__(self)
        self.title = title
        self.ops : List[UnitaryOperation] = []
    def append_op(self, op : UnitaryOperation) -> None:
        self.ops.append(op)
    def reverse(self) -> None:
        for op in reversed(self.ops):
            op.reverse()
        self.ops.reverse()
    def synthesis(self, bq_circuit : blueqat.Circuit) -> blueqat.Circuit:
        for op in self.ops:
            op.synthesis(bq_circuit)
        return bq_circuit
    def string(self, depth: int) -> str:
        s = self.title + '{\n'
        s += ',\n'.join(
            [' ' * (depth + 1) + op.string(depth + 1) for op in self.ops])
        s += '\n'
        s += ' ' * depth + '}'
        return s
class UO_H(UnitaryOperation):
    def __init__(self, q : int) -> None:
        UnitaryOperation.__init__(self)
        self.q = q
        append_uo(self)
    def reverse(self) -> None:
        pass
    def synthesis(self, bq_circuit : blueqat.Circuit) -> blueqat.Circuit:
        bq_circuit.h[self.q]
        return bq_circuit
    def string(self, depth: int) -> str:
        return 'H[{0}]'.format(str(self.q))
class UO_X(UnitaryOperation):
    def __init__(self, q: int) -> None:
        UnitaryOperation.__init__(self)
        self.q = q
        append_uo(self)
    def reverse(self) -> None:
        pass
    def synthesis(self, bq_circuit : blueqat.Circuit) -> blueqat.Circuit:
        bq_circuit.x[self.q]
        return bq_circuit
    def string(self, depth: int) -> str:
        return 'X[{0}]'.format(str(self.q))
class UO_CX(UnitaryOperation):
    def __init__(self, c: int, x: int) -> None:
        UnitaryOperation.__init__(self)
        self.c = c
        self.x = x
        append_uo(self)
    def reverse(self) -> None:
        pass
    def synthesis(self, bq_circuit: blueqat.Circuit) -> blueqat.Circuit:
        bq_circuit.cx[self.c, self.x]
        return bq_circuit
    def string(self, depth: int) -> str:
        return 'CX[{0},{1}]'.format(str(self.c), str(self.x))
class UO_CSWAP(UnitaryOperation):
    def __init__(self, c: int, a: int, b: int) -> None:
        UnitaryOperation.__init__(self)
        self.c = c
        self.a = a
        self.b = b
        append_uo(self)
    def reverse(self) -> None:
        pass
    def synthesis(self, bq_circuit: blueqat.Circuit) -> blueqat.Circuit:
        (c, a, b) = (self.c, self.a, self.b)
        bq_circuit.ccx[c, a, b].ccx[c, b, a].ccx[c, a, b]
        return bq_circuit
    def string(self, depth: int) -> str:
        return 'CSWAP[{0},{1},{2}]'.format(
            str(self.c), str(self.a), str(self.b))
class QBit(object):
    def __init__(self, index: Optional[int] = None) -> None:
        self.index : int = (index if index is not None else
                            ALLOCATOR.allocate1())
    def deallocate(self) -> int:
        ALLOCATOR.deallocate1(self.index)
        return self.index
    @staticmethod
    def h(q: 'QBit') -> None:
        UO_H(q.index)
    @staticmethod
    def x(q: 'QBit') -> None:
        UO_X(q.index)
    @staticmethod
    def cx(c: 'QBit', q: 'QBit') -> None:
        UO_CX(c.index, q.index)
    @staticmethod
    def cswap(c: 'QBit', a: 'QBit', b: 'QBit') -> None:
        UO_CSWAP(c.index, a.index, b.index)
    def parse(self, s: str) -> int:
        i = s.find("'")
        s = s[i + 1:]
        return 0 if s[self.index] == '0' else 1
    def __str__(self) -> str:
        return str(self.index)
# ----------------------------------------------------------------------
PROCEDURE_STACK_TOP : Optional['UO_Proc'] = None
class UO_Proc(object):
    def __init__(self, title: str) -> None:
        self._procedure = UO_Procedure(title)
    def __enter__(self) -> 'UO_Proc':
        global PROCEDURE_STACK_TOP
        self._parent = PROCEDURE_STACK_TOP
        PROCEDURE_STACK_TOP = self
        return self
    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        global PROCEDURE_STACK_TOP
        PROCEDURE_STACK_TOP = self._parent
        if PROCEDURE_STACK_TOP:
            PROCEDURE_STACK_TOP.get_procedure().append_op(
                self._procedure)
    def __str__(self) -> str:
        return str(self._procedure)
    def get_procedure(self) -> 'UO_Procedure':
        return self._procedure
    def synthesis(self, bq_circuit: blueqat.Circuit) -> blueqat.Circuit:
        return self._procedure.synthesis(bq_circuit)
def dump_uo_stack() -> None:
    if PROCEDURE_STACK_TOP is None:
        raise RuntimeError('procedure not started')
    p : Optional[UO_Proc] = PROCEDURE_STACK_TOP
    while p is not None:
        print(str(p.get_procedure()))
        p = p._parent
def append_uo(uo : 'UnitaryOperation') -> None:
    if PROCEDURE_STACK_TOP is None:
        raise RuntimeError('procedure not started')
    PROCEDURE_STACK_TOP.get_procedure().append_op(uo)
def reverse_current_proc() -> None:
    if PROCEDURE_STACK_TOP is None:
        raise RuntimeError('procedure not started')
    PROCEDURE_STACK_TOP.get_procedure().reverse()
# ----------------------------------------------------------------------
class Carry(UnitaryOperation):
    def __init__(self,
                 a: QBit, b: QBit, c: QBit, d: QBit,
                 is_reversed: bool=False) -> None:
        UnitaryOperation.__init__(self)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        append_uo(self)
        self.is_reversed = is_reversed
    def reverse(self) -> None:
        self.is_reversed = not self.is_reversed
    def synthesis(self, bq_circuit: blueqat.Circuit) -> blueqat.Circuit:
        (a, b, c, d) = (self.a.index, self.b.index, self.c.index, self.d.index)
        if not self.is_reversed:
            bq_circuit.ccx[b, c, d].cx[b, c].ccx[a, c, d]
        else:
            bq_circuit.ccx[a, c, d].cx[b, c].ccx[b, c, d]
        return bq_circuit
    def string(self, depth: int) -> str:
        return (
            'Carry[{0},{1},{2},{3}]{{ ccx[{1},{2},{3}], cx[{1},{2}], ccx[{0},{2},{3}] }}'.format(str(self.a), str(self.b), str(self.c), str(self.d))
            if not self.is_reversed else
            'RCarry[{0},{1},{2},{3}]{{ ccx[{0},{2},{3}], cx[{1},{2}], ccx[{1},{2},{3}] }}'.format(str(self.a), str(self.b), str(self.c), str(self.d)))
class Sum(UnitaryOperation):
    def __init__(self,
                 a: QBit, b: QBit, c: QBit,
                 is_reversed: bool = False) -> None:
        UnitaryOperation.__init__(self)
        self.a = a
        self.b = b
        self.c = c
        append_uo(self)
        self.is_reversed = is_reversed
    def reverse(self) -> None:
        self.is_reversed = not self.is_reversed
    def synthesis(self, bq_circuit: blueqat.Circuit) -> blueqat.Circuit:
        (a, b, c) = (self.a.index, self.b.index, self.c.index)
        if not self.is_reversed:
            bq_circuit.cx[b, c].cx[a, c]
        else:
            bq_circuit.cx[a, c].cx[b, c]
        return bq_circuit
    def string(self, depth: int) -> str:
        return (
            'Sum[{0},{1},{2}]{{ cx[{1},{2}], cx[{0},{2}] }}'.format(
                str(self.a), str(self.b), str(self.c))
            if not self.is_reversed else
            'RSum[{0},{1},{2}]{{ cx[{0},{2}], cx[{1},{2}] }}'.format(
                str(self.a), str(self.b), str(self.c)))
class Integer(object):
    NBITS = 4
    def __init__(self, n: int) -> None:
        nbits = Integer.NBITS
        self.qbits = [QBit() for _ in range(nbits)]
        self.cs : Optional[List[QBit]] = None
        self._carry : Optional[QBit] = None
        if n != 0:
            with UO_Proc('Integer.init') as p:
                for i in range(0, nbits):
                    if n & (1 << i) == 0:
                        continue
                    QBit.x(self.qbits[i])
    def deallocate(self) -> None:
        [c.deallocate() for c in self.qbits]
    def _deallocate(self) -> None:
        if self.cs:
            [c.deallocate() for c in self.cs]
            self.cs = None
    def _cs(self, i: int) -> QBit:
        if i >= len(self.qbits):
            return self.carry()
        if not self.cs:
            self.cs = [QBit() for _ in range(len(self.qbits))]
        return self.cs[i]
    def carry(self) -> QBit:
        if self._carry is None:
            self._carry = QBit()
        return self._carry
    def __iadd__(self, other: 'Integer') -> 'Integer':
        self._synthesis_iadd('Integer.add', other)
        self._deallocate()
        return self
    def __isub__(self, other: 'Integer') -> 'Integer':
        self._synthesis_iadd('Integer.sub', other,
                             is_reversed = True)
        self._deallocate()
        return self
    def _synthesis_iadd(self, title: str, other: 'Integer',
                        is_reversed : bool = False) -> None:
        if len(other.qbits) != len(self.qbits):
            raise RuntimeError('invalid Integer bit width')
        with UO_Proc(title):
            for i in range(len(self.qbits)):
                a = other.qbits[i]
                b = self.qbits[i]
                c = self._cs(i)
                c2 = self._cs(i + 1)
                Carry(c, a, b, c2)
            QBit.cx(other.qbits[-1], self.qbits[-1])
            for i in range(len(self.qbits) - 1, 0, -1):
                a = other.qbits[i]
                b = self.qbits[i]
                c = self._cs(i)
                a1 = other.qbits[i - 1]
                b1 = self.qbits[i - 1]
                c1 = self._cs(i - 1)
                Sum(c, a, b)
                Carry(c1, a1, b1, c, is_reversed=True)
            Sum(self._cs(0), other.qbits[0], self.qbits[0])
            if is_reversed:
                reverse_current_proc()
    def __lshift__(self, orig: 'Integer') -> 'Integer':
        if len(orig.qbits) != len(self.qbits):
            raise RuntimeError('invalid Integer bit width')
        with UO_Proc('Integer.xor'):
            for c, x in zip(orig.qbits, self.qbits):
                QBit.cx(c, x)
            if orig._carry is not None:
                QBit.cx(orig._carry, self.carry())
        return self
    def hadamard(self, n: int = -1) -> 'Integer':
        with UO_Proc('Integer.hadamard'):
            if n < 0:
                n = len(self.qbits)
            for i in range(0, n):
                QBit.h(self.qbits[i])
        return self
    @staticmethod
    def cswap(c: QBit, a: 'Integer', b: 'Integer') -> None:
        with UO_Proc('Integer.cswap'):
            if len(a.qbits) != len(b.qbits):
                raise RuntimeError('invalid Integer bit width')
            for a1, b1 in zip(a.qbits, b.qbits):
                QBit.cswap(c, a1, b1)
            if a._carry is not None:
                QBit.cswap(c, a._carry, b.carry())
            elif b._carry is not None:
                QBit.cswap(c, a.carry(), b._carry)
    def bit_indices(self) -> List[int]:
        bs = []
        for i in range(0, len(self.qbits)):
            bs.append(self.qbits[i].index)
        if self._carry is not None:
            bs.append(self._carry.index)
        return bs
    def parse(self, s: str) -> int:
        i = s.find("'")
        s = s[i + 1:]
        n = 0
        for index in reversed(self.bit_indices()):
            n = n * 2
            if s[index] == '1':
                n += 1
        return n
    def parse_signed(self, s: str) -> int:
        n = self.parse(s)
        if n > (1 << len(self.qbits)):
            return n - (1 << (len(self.qbits) + 1))
        return n
    def __str__(self) -> str:
        args = ','.join([str(index) for index in self.bit_indices()])
        return 'Integer[{0}]'.format(args)
# ----------------------------------------------------------------------
class TestProc_ADD(UO_Proc):
    def __init__(self) -> None:
        UO_Proc.__init__(self, 'TestProc_ADD')
    def __enter__(self) -> 'TestProc_ADD':
        UO_Proc.__enter__(self)
        Integer.NBITS = 4
        a = Integer(0)
        b = Integer(0)
        a.hadamard()
        b.hadamard()
        b0 = Integer(0)
        b0 << b
        b += a
        self.a = a
        self.b0 = b0
        self.b = b
        return self
with TestProc_ADD() as p_add:
    print('---- circuit ----')
    print(str(p_add))
    print('a =' + str(p_add.a))
    print('b0=' + str(p_add.b0))
    print('b =' + str(p_add.b))
    print('---- simulator ----')
    circuit = p_add.synthesis(blueqat.Circuit())
    print(str(circuit))
    print('---- result ----')
    start = time.perf_counter()
    for _ in range(10):
        result = str(circuit.m[:].run(shots=1))
        a = p_add.a.parse(result)
        b0 = p_add.b0.parse(result)
        b = p_add.b.parse(result)
        if a + b0 == b:
            print(str(result) + ' OK {0}+{1}={2}'.format(a, b0, b))
        else:
            print(str(result) + ' NG {0}+{1}={2}'.format(a, b0, b))
            sys.exit(1)
    end = time.perf_counter()
    print('TIME_ADD: {0:.3f} sec'.format(end-start))
reset_all()

# ----------------------------------------------------------------------
class TestProc_SUB(UO_Proc):
    def __init__(self) -> None:
        UO_Proc.__init__(self, 'TestProc_SUB')
    def __enter__(self) -> 'TestProc_SUB':
        UO_Proc.__enter__(self)
        Integer.NBITS = 4
        a = Integer(0)
        b = Integer(0)
        a.hadamard()
        b.hadamard()
        b0 = Integer(0)
        b0 << b
        b -= a
        self.a = a
        self.b0 = b0
        self.b = b
        return self
with TestProc_SUB() as p_sub:
    print('---- circuit ----')
    print(str(p_sub))
    print('a =' + str(p_sub.a))
    print('b0=' + str(p_sub.b0))
    print('b =' + str(p_sub.b))
    print('---- simulator ----')
    circuit = p_sub.synthesis(blueqat.Circuit())
    print(str(circuit))
    print('---- result ----')
    start = time.perf_counter()
    for _ in range(10):
        result = str(circuit.m[:].run(shots=1))
        a = p_sub.a.parse(result)
        b0 = p_sub.b0.parse(result)
        b = p_sub.b.parse_signed(result)
        if b0 -a == b:
            print(str(result) + ' OK {0}-{1}={2}'.format(b0, a, b))
        else:
            print(str(result) + ' NG {0}-{1}={2}'.format(b0, a, b))
            sys.exit(1)
    end = time.perf_counter()
    print('TIME_SUB: {0:.3f} sec'.format(end-start))
reset_all()

# ----------------------------------------------------------------------
class TestProc_MODULOADD(UO_Proc):
    def __init__(self) -> None:
        UO_Proc.__init__(self, 'MODULOADD')
    def __enter__(self) -> 'TestProc_MODULOADD':
        UO_Proc.__enter__(self)
        Integer.NBITS = 3
        with UO_Proc('MODULOADD.init'):
            a = Integer(0)
            a.hadamard(2)
            b = Integer(0)
            b.hadamard(2)
            n = Integer(4)
            n.hadamard(2)
            b0 = Integer(0)
            b0 << b
        with UO_Proc('MODULOADD.main'):
            b += a
            b -= n
            flag = QBit()
            QBit.cx(b.carry(), flag)
            n0 = Integer(0)
            Integer.cswap(flag, n, n0)
            b += n0
            Integer.cswap(flag, n, n0)
            n0.deallocate()
        with UO_Proc('MODULOADD_freeflag'):
            b -= a
            QBit.x(b.carry())
            QBit.cx(b.carry(), flag)
            QBit.x(b.carry())
            b += a
            flag.deallocate()
        self.a = a
        self.b0 = b0
        self.b = b
        self.n = n
        self.flag = flag
        return self
with TestProc_MODULOADD() as p:
    print('---- circuit ----')
    print(str(p))
    print('a =' + str(p.a))
    print('b0=' + str(p.b0))
    print('n =' + str(p.n))
    print('b =' + str(p.b))
    print('---- simulator ----')
    circuit = p.synthesis(blueqat.Circuit())
    print(str(circuit))
    print('---- result ----')
    start = time.perf_counter()
    for _ in range(10):
        result = str(circuit.m[:].run(shots=1))
        a = p.a.parse(result)
        b0 = p.b0.parse(result)
        n = p.n.parse(result)
        b = p.b.parse(result)
        if ((n == 0 and a + b0 == b) or
            ((a + b0) - b) % n == 0):
            print(str(result) + ' OK {0}+{1} mod {2} = {3}'.format(
                a, b0, n, b))
        else:
            print(str(result) + ' NG {0}+{1} mod {2} = {3}'.format(
                a, b0, n, b))
            sys.exit(1)
        if p.flag.parse(result) != 0:
            print('FLAG IS DIRTY')
            sys.exit(1)
    end = time.perf_counter()
    print('TIME_MODULOADD: {0:.3f} sec'.format(end-start))
