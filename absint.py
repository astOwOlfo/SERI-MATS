from typing import Callable, Any
from sys import float_info
from queue import PriorityQueue
from tqdm import tqdm
from dataclasses import dataclass, field

class Interval:
    def __init__(self, min: float, max: float):
        self.min = float(min)
        self.max = float(max)
        assert self.min <= self.max

    def singleton(x: float):
        return Interval(x, x)

    def wrap(i, j):
        return Interval(min(i.min, j.min), max(i.max, j.max))

    def account_for_rounding_errors(self):
        return Interval( self.min - abs(self.min) * float_info.epsilon,
                         self.max + abs(self.max) * float_info.epsilon )

    def __add__(self, other):
        return Interval(self.min + other.min, self.max + other.max).account_for_rounding_errors()

    def __sub__(self, other):
        return Interval(self.min - other.max, self.max - other.min).account_for_rounding_errors()

    def __mul__(self, other):
        possibilities = [a * b for a in (self.min, self.max) for b in (other.min, other.max)]
        return Interval(min(possibilities), max(possibilities)).account_for_rounding_errors()
    
    def can_be_nonnegative(self):
        return self.max >= 0
    
    def can_be_negative(self):
        return self.min < 0

    def length(self):
        return Interval.singleton(self.max - self.min).account_for_rounding_errors()

    def round(self):
        return (self.min + self.max) / 2

    def split(self):
        pivot = (self.min + self.max) / 2
        return Interval(self.min, pivot), Interval(pivot, self.max)

class Box2D:
    def __init__(self, x: Interval, y: Interval):
        self.x = x
        self.y = y

    def area(self) -> Interval:
        return self.x.length() * self.y.length()
    
    def split(self):
        x1, x2 = self.x.split()
        y1, y2 = self.y.split()
        return Box2D(x1, y1), Box2D(x1, y2), Box2D(x2, y1), Box2D(x2, y2)

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)

def probability_nonnegative(f: Callable[[Box2D], Interval], domain: Box2D, iterations: int) -> Interval:
    # Before the loop and after each iteration, the following invariant is maintained:
    # - `nonnegative_on` is a list of boxes such that `f` is nonnegative on each one of them
    # - if f(x) is non negative for some x in domain, then x is contained in a box either in
    #     `nonnegative_on` or in `worklist``
    # - the elementsofnonnegative_on and worklist are pairwise disjoint and contained in `domain`

    nonnegative_on = []
    worklist = PriorityQueue()
    worklist.put(PrioritizedItem(-domain.area().round(), domain))
    
    for _ in tqdm(range(iterations)):
        if worklist.empty():
            break

        box = worklist.get().item
        f_on_box = f(box)
        
        if not f_on_box.can_be_negative():
            nonnegative_on.append(box)
        
        if f_on_box.can_be_negative() and f_on_box.can_be_nonnegative():
            for quarter in box.split():
                worklist.put(PrioritizedItem(-quarter.area().round(), quarter))

    lower_bound = sum( (box.area() for box in nonnegative_on),
                        start=Interval.singleton(0) )
    upper_bound = lower_bound + sum( (box.item.area() for box in worklist.queue),
                                     start=Interval.singleton(0) )
    return Interval.wrap(lower_bound, upper_bound)

def approximate_pi(iterations):
    # We have `pi = 4 * Prob(f(u) >= 0 | u \in [0, 1] x [0, 1])`.
    # We approximate `pi` by approximating this quantity.
    def f(vector: Box2D):
        return Interval.singleton(1) - vector.x * vector.x - vector.y * vector.y
    domain = Box2D(Interval(0, 1), Interval(0, 1))
    pi_approx = Interval.singleton(4) * probability_nonnegative(f, domain, iterations=iterations)
    return pi_approx

if __name__ == "__main__":
    pi_approx = approximate_pi(iterations=100_000)
    print(f"Pi is between {pi_approx.min} and {pi_approx.max}.")
