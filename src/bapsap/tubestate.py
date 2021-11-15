from .state_space import StateNode
from itertools import product, combinations


class TubeState(StateNode):
    """A Game State Class for the state of the tubes in the ball sort Game."""

    def __init__(self, tubes=None, tube_size=4, predecessor=None, action=None) -> None:
        super().__init__(predecessor, action)
        self.tubes = tubes
        self.tube_size = tube_size
        self.search_strategies.update(
            {"sahc_with_neg_state_entropy": self.sahc_with_negative_state_entropy}
        )

    @staticmethod
    def all_same(list_):
        """Checks if all the values in the list are same. Also returns true if empty"""
        return len(set(list_)) <= 1

    def valid_actions(self):
        """Generates the possible next moves for the current tube state"""
        empty_tubes = {i for i in range(len(self.tubes)) if len(self.tubes[i]) == 0}
        filled_tubes = set(range(len(self.tubes))) - empty_tubes
        for f, t in combinations(filled_tubes, 2):
            if self.tubes[f][-1] == self.tubes[t][-1]:
                for f1, t1 in ((f, t), (t, f)):
                    if len(self.tubes[t1]) < self.tube_size:
                        c = -1
                        try:
                            while self.tubes[f1][c - 1] == self.tubes[f1][-1]:
                                c -= 1
                        except IndexError:
                            pass
                        if self.tube_size - len(self.tubes[t1]) + c < 0:
                            continue
                        yield (f1, t1)
        if empty_tubes:
            for f, t in product(filled_tubes, {empty_tubes.pop()}):
                if not TubeState.all_same(self.tubes[f]):
                    yield (f, t)

    def is_action_valid(self, action):
        """Checks whether the given action is valid."""
        f, t = action
        try:
            f_tube, t_tube = self.tubes[f], self.tubes[t]
        except (IndexError):
            return False
        # to tube empty or
        # to tube is not full and top balls in from tube and to tube are same
        return (
            len(t_tube) == 0
            or len(t_tube) < self.tube_size
            and f_tube[-1] == t_tube[-1]
        )

    def do_action(self, action):
        """Performs the action on the object."""
        f, t = action
        # if empty action do nothing
        if f is None and t is None:
            return self
        # else check if action is possible and do the action
        if not self.is_action_valid(action):
            raise ValueError("Cannot do action. Action Invalid.")
        self.tubes[t].append(self.tubes[f].pop())
        return self

    def undo_action(self, action):
        """Undos the given action and reverts back to the previous state."""
        f, t = action
        self.tubes[f].append(self.tubes[t].pop())
        return self

    def is_goal(self):
        """Checks whether the tube state is in solved state."""
        return all(map(TubeState.all_same, self.tubes))

    @staticmethod
    def tube_entropy(tube):
        entropy = 0
        if tube:
            # entropy atleast one for non empty tubes
            entropy += 1
        for i in range(1, len(tube)):
            # add one to entropy for every change of color
            if tube[i] != tube[i - 1]:
                entropy += 1
        return entropy

    def state_entropy(self):
        """Returns the amount of disorder in the state"""
        return sum([TubeState.tube_entropy(tube) for tube in self.tubes])

    def _solution_finishup(self, solution):
        """
        Add the actions required to go from the all-same state to
        all-in-one state to complete the solution.
        """
        for action in solution:
            self.do_action(action)

        partially_filled_tube_indices = filter(
            lambda i: len(self.tubes[i]) != 0 != self.tube_size, range(len(self.tubes))
        )

        for i, j in combinations(partially_filled_tube_indices, 2):
            i_tube, j_tube = self.tubes[i], self.tubes[j]
            if i_tube and j_tube and i_tube[-1] == j_tube[-1]:
                i_len, j_len = len(i_tube), len(j_tube)
                if i_len < j_len:
                    solution.extend([(i, j)] * i_len)
                else:
                    solution.extend([(j, i)] * j_len)
        return solution

    def find_solution(self, strategy="dfs"):
        solution = super().find_solution(strategy=strategy)
        return self._solution_finishup(solution)

    def sahc_with_negative_state_entropy(self):
        return self.sahc(heuristic_func=lambda x: -TubeState.state_entropy(x))

    def __str__(self):
        return str(self.tubes)

    def __hash__(self) -> int:
        # the hash function to be used when used with sets
        return hash(tuple(map(tuple, self.tubes)))

    def __eq__(self, o: object) -> bool:
        # the eq function to be used when used with sets
        return self.tubes == o.tubes
