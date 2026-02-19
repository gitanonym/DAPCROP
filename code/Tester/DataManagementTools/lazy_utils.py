from typing import Callable, Generator

"""
This module contains the generic lazy_product function.
The function generates the cartesian product of the given generators.
The function is lazy and generates the product on the fly, compared to itertools.product which generates the product at once.

Example:
    def gen1():
        for i in range(2):
            print(i)
            yield i

    def gen2():
        for i in range(2, 4):
            print(i)
            yield i

    generators = [gen1, gen2]
    for product in lazy_product(generators):
        print(product)
    # Output:
    # 0
    # 2
    # [0, 2]
    # 3
    # [0, 3]
    # 1
    # 2
    # [1, 2]
    # 3
    # [1, 3]
    
    # Compared output of itertools.product:
    # 0
    # 1
    # 2
    # 3
    # (0, 2)
    # (0, 3)
    # (1, 2)
    # (1, 3)
"""

SINGLE_ARGUMENT_ALL_GENERATORS = 0
MULTIPLE_ARGUMENTS_ALL_GENERATORS = 1
SINGLE_ARGUMENT_EACH_GENERATOR = 2
MULTIPLE_ARGUMENTS_EACH_GENERATOR = 3

def lazy_product(
        generators: list[Callable[[], Generator[object, object, None]]], 
        args_lists: object | list[object] | list[list[object]] = [],
        args_type: int = 1,
        current_product=[]
 ) -> Generator[list[object], None, None]:
    """
    Generate the cartesian product of the given generators.
    The function is lazy and generates the product on the fly.

    Parameters:
        generators (list[Callable[[], Generator[object, object, None]]): List of generators.
        args_list (object | list[object] | list[list[object]]): List of arguments for the generators.
        args_type (int): The type of the args_list.
            0 - Single object for ALL generators
            1 - List of arguments for ALL generators
            2 - List of single objects for EACH generator
            3 - List of list of arguments for EACH generator
        current_product (list[object]): The current product.
    """
    if not generators:
        yield current_product
        return

    args_manipulator = {
        SINGLE_ARGUMENT_ALL_GENERATORS: lambda x: [[x] for _ in range(len(generators))], #   Single object for ALL generators - wrap in list for unpacking
        MULTIPLE_ARGUMENTS_ALL_GENERATORS: lambda x: [x for _ in range(len(generators))], # List of arguments for ALL generators - duplicate for each generator
        SINGLE_ARGUMENT_EACH_GENERATOR: lambda x: [[item] for item in x], # List of single objects for EACH generator, wrap each single in list for unpacking
        MULTIPLE_ARGUMENTS_EACH_GENERATOR: lambda x: x, #   List of list of arguments for EACH generator, no need to manipulate
    }

    args_lists = args_manipulator[args_type](args_lists)
    assert len(generators) == len(args_lists), "Generators and args_lists must have the same length"

    for item in generators[0](*(args_lists[0])):
        yield from lazy_product(generators[1:], args_lists[1: ], 3, current_product + [item])
