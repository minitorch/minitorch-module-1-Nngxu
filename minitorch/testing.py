# type: ignore

from typing import Callable, Generic, Iterable, Tuple, TypeVar

import minitorch.operators as operators

A = TypeVar("A")


class MathTest(Generic[A]):
    @staticmethod
    def neg(a: A) -> A:
        "Negate the argument"
        return -a

    @staticmethod
    def addConstant(a: A) -> A:
        "Add contant to the argument"
        return 5 + a

    @staticmethod
    def square(a: A) -> A:
        "Manual square"
        return a * a

    @staticmethod
    def cube(a: A) -> A:
        "Manual cube"
        return a * a * a

    @staticmethod
    def subConstant(a: A) -> A:
        "Subtract a constant from the argument"
        return a - 5

    @staticmethod
    def multConstant(a: A) -> A:
        "Multiply a constant to the argument"
        return 5 * a

    @staticmethod
    def div(a: A) -> A:
        "Divide by a constant"
        return a / 5

    @staticmethod
    def inv(a: A) -> A:
        "Invert after adding"
        return operators.inv(a + 3.5)

    @staticmethod
    def sig(a: A) -> A:
        "Apply sigmoid"
        return operators.sigmoid(a)

    @staticmethod
    def log(a: A) -> A:
        "Apply log to a large value"
        return operators.log(a + 100000)

    @staticmethod
    def relu(a: A) -> A:
        "Apply relu"
        return operators.relu(a + 5.5)

    @staticmethod
    def exp(a: A) -> A:
        "Apply exp to a smaller value"
        return operators.exp(a - 200)

    @staticmethod
    def explog(a: A) -> A:
        return operators.log(a + 100000) + operators.exp(a - 200)

    @staticmethod
    def add2(a: A, b: A) -> A:
        "Add two arguments"
        return a + b

    @staticmethod
    def mul2(a: A, b: A) -> A:
        "Mul two arguments"
        return a * b

    @staticmethod
    def div2(a: A, b: A) -> A:
        "Divide two arguments"
        return a / (b + 5.5)

    @staticmethod
    def gt2(a: A, b: A) -> A:
        return operators.lt(b, a + 1.2)

    @staticmethod
    def lt2(a: A, b: A) -> A:
        return operators.lt(a + 1.2, b)

    @staticmethod
    def eq2(a: A, b: A) -> A:
        return operators.eq(a, (b + 5.5))

    @staticmethod
    def sum_red(a: Iterable[A]) -> A:
        return operators.sum(a)

    @staticmethod
    # red means reduce
    def mean_red(a: Iterable[A]) -> A:
        return operators.sum(a) / float(len(a))

    @staticmethod
    def mean_full_red(a: Iterable[A]) -> A:
        return operators.sum(a) / float(len(a))

    @staticmethod
    def complex(a: A) -> A:
        return (
            operators.log(
                operators.sigmoid(
                    operators.relu(operators.relu(a * 10 + 7) * 6 + 5) * 10
                )
            )
            / 50
        )

    @classmethod
    def _tests(
        cls,
    ) -> Tuple[
        Tuple[str, Callable[[A], A]],
        Tuple[str, Callable[[A, A], A]],
        Tuple[str, Callable[[Iterable[A]], A]],
    ]:
        """
        Returns a list of all the math tests.
        """
        one_arg = []
        two_arg = []
        red_arg = []
        for k in dir(MathTest): # dir(MathTest) returns a list of all the attributes name(str type) in the MathTest class
            if callable(getattr(MathTest, k)) and not k.startswith("_"):
                # base_fn is <function MathTest.xxx at xxxxxx>
                base_fn = getattr(cls, k)
                # scalar_fn = getattr(cls, k)
                tup = (k, base_fn) # tup is like ('add2', <function MathTest.add2 at 0x7f8ad8d18820>)
                if k.endswith("2"):
                    two_arg.append(tup)
                elif k.endswith("red"):
                    red_arg.append(tup)
                else:
                    one_arg.append(tup)
        return one_arg, two_arg, red_arg

    @classmethod
    def _comp_testing(cls):
        # cls._tests() return
        # ([('addConstant', <function MathTest.addConstant at 0x7f49121ef160>),
        #   ('complex', <function MathTestVariable.complex at 0x7f49121f05e0>),
        #   ('cube', <function MathTest.cube at 0x7f49121ef280>),
        #   ('div', <function MathTest.div at 0x7f49121ef430>),
        #   ('exp', <function MathTestVariable.exp at 0x7f49121f0160>),
        #   ('explog', <function MathTestVariable.explog at 0x7f49121f01f0>),
        #   ('inv', <function MathTestVariable.inv at 0x7f49121efee0>),
        #   ('log', <function MathTestVariable.log at 0x7f49121f0040>),
        #   ('multConstant', <function MathTest.multConstant at 0x7f49121ef3a0>),
        #   ('neg', <function MathTest.neg at 0x7f49121ef0d0>),
        #   ('relu', <function MathTestVariable.relu at 0x7f49121f00d0>),
        #   ('sig', <function MathTestVariable.sig at 0x7f49121eff70>),
        #   ('square', <function MathTest.square at 0x7f49121ef1f0>),
        #   ('subConstant', <function MathTest.subConstant at 0x7f49121ef310>)],
        #  [('add2', <function MathTest.add2 at 0x7f49121ef820>),
        #   ('div2', <function MathTest.div2 at 0x7f49121ef940>),
        #   ('eq2', <function MathTestVariable.eq2 at 0x7f49121f0430>),
        #   ('gt2', <function MathTestVariable.gt2 at 0x7f49121f04c0>),
        #   ('lt2', <function MathTestVariable.lt2 at 0x7f49121f0550>),
        #   ('mul2', <function MathTest.mul2 at 0x7f49121ef8b0>)],
        #  [('mean_full_red',
        #    <function MathTestVariable.mean_full_red at 0x7f49121f03a0>),
        #   ('mean_red', <function MathTestVariable.mean_red at 0x7f49121f0310>),
        #   ('sum_red', <function MathTestVariable.sum_red at 0x7f49121f0280>)])
        one_arg, two_arg, red_arg = cls._tests()
        # MathTest._tests() return
        # ([('addConstant', <function MathTest.addConstant at 0x7fe6821ef160>),
        #   ('complex', <function MathTest.complex at 0x7fe6821efd30>),
        #   ('cube', <function MathTest.cube at 0x7fe6821ef280>),
        #   ('div', <function MathTest.div at 0x7fe6821ef430>),
        #   ('exp', <function MathTest.exp at 0x7fe6821ef700>),
        #   ('explog', <function MathTest.explog at 0x7fe6821ef790>),
        #   ('inv', <function MathTest.inv at 0x7fe6821ef4c0>),
        #   ('log', <function MathTest.log at 0x7fe6821ef5e0>),
        #   ('multConstant', <function MathTest.multConstant at 0x7fe6821ef3a0>),
        #   ('neg', <function MathTest.neg at 0x7fe6821ef0d0>),
        #   ('relu', <function MathTest.relu at 0x7fe6821ef670>),
        #   ('sig', <function MathTest.sig at 0x7fe6821ef550>),
        #   ('square', <function MathTest.square at 0x7fe6821ef1f0>),
        #   ('subConstant', <function MathTest.subConstant at 0x7fe6821ef310>)],
        #  [('add2', <function MathTest.add2 at 0x7fe6821ef820>),
        #   ('div2', <function MathTest.div2 at 0x7fe6821ef940>),
        #   ('eq2', <function MathTest.eq2 at 0x7fe6821efaf0>),
        #   ('gt2', <function MathTest.gt2 at 0x7fe6821ef9d0>),
        #   ('lt2', <function MathTest.lt2 at 0x7fe6821efa60>),
        #   ('mul2', <function MathTest.mul2 at 0x7fe6821ef8b0>)],
        #  [('mean_full_red', <function MathTest.mean_full_red at 0x7fe6821efca0>),
        #   ('mean_red', <function MathTest.mean_red at 0x7fe6821efc10>),
        #   ('sum_red', <function MathTest.sum_red at 0x7fe6821efb80>)])
        one_argv, two_argv, red_argv = MathTest._tests()
        # one_arg return
        # [('addConstant',
        #   <function MathTest.addConstant at 0x7fe6821ef160>,
        #   <function MathTest.addConstant at 0x7fe6821ef160>),
        #  ('complex',
        #   <function MathTest.complex at 0x7fe6821efd30>,
        #   <function MathTestVariable.complex at 0x7fe6821f05e0>),
        #  ...
        # ]
        one_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(one_arg, one_argv)]
        # two_arg return
        # [('addConstant',
        #   <function MathTest.addConstant at 0x7fe6821ef160>,
        #   <function MathTest.addConstant at 0x7fe6821ef160>),
        #   ...
        # ]
        two_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(two_arg, two_argv)]
        red_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(red_arg, red_argv)]
        return one_arg, two_arg, red_arg


class MathTestVariable(MathTest):
    @staticmethod
    def inv(a):
        return 1.0 / (a + 3.5)

    @staticmethod
    def sig(x):
        return x.sigmoid()

    @staticmethod
    def log(x):
        return (x + 100000).log()

    @staticmethod
    def relu(x):
        return (x + 5.5).relu()

    @staticmethod
    def exp(a):
        return (a - 200).exp()

    @staticmethod
    def explog(a):
        return (a + 100000).log() + (a - 200).exp()

    @staticmethod
    def sum_red(a):
        return a.sum(0)

    @staticmethod
    def mean_red(a):
        return a.mean(0)

    @staticmethod
    def mean_full_red(a):
        return a.mean()

    @staticmethod
    def eq2(a, b):
        return a == (b + 5.5)

    @staticmethod
    def gt2(a, b):
        return a + 1.2 > b

    @staticmethod
    def lt2(a, b):
        return a + 1.2 < b

    @staticmethod
    def complex(a):
        return (((a * 10 + 7).relu() * 6 + 5).relu() * 10).sigmoid().log() / 50
