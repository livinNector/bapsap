from abc import ABC, abstractmethod
from collections import deque
import copy


class StateNode(ABC):
    def __init__(self, predecessor=None, action=None):
        self.predecessor = predecessor
        self.action = action
        # the values that are not deep copied during state copy
        self.reference_variables = ["predecessor"]
        self.search_strategies = {"bfs": self.bfs, "dfs": self.dfs}

    @abstractmethod
    def do_action(self, action):
        pass

    @abstractmethod
    def undo_action(self, action):
        pass

    def state_copy(self):
        """Returns a deep copy.copy of state variables without affecting predecessor reference"""
        copy_ = copy.copy(self)
        copy_.__dict__ = {
            attr: (
                copy.deepcopy(self.__dict__[attr])
                if attr not in self.reference_variables
                else self.__dict__[attr]
            )
            for attr in self.__dict__
        }
        return copy_

    @abstractmethod
    def is_action_valid(self):
        pass

    @abstractmethod
    def valid_actions(self):
        """Generates the valid actions from the current state."""
        pass

    @abstractmethod
    def is_goal(self):
        """Checks whether the state is the goal state"""
        pass

    def successors(self):
        """Generates the successors of the current state."""
        for action in self.valid_actions():
            successor = self.state_copy()
            successor.do_action(action)
            successor.action = action
            successor.predecessor = self
            yield successor

    def traceback_path(self):
        path = deque()
        current_state = self
        while current_state.predecessor is not None:
            path.appendleft(current_state.action)
            current_state = current_state.predecessor
        return path

    def bfs(self):
        """Breadth First Search"""
        fringe = deque([self])
        visited = set([self])
        while fringe:
            current_state = fringe.pop()
            if current_state.is_goal():
                print("visited: ", len(visited))
                return current_state.traceback_path()
            successors = set(
                filter(lambda state: state not in visited, current_state.successors())
            )
            fringe.extendleft(successors)
            visited.update(successors)

    def dfs(self):
        """Depth First Search."""
        fringe = deque([self])
        visited = set([self])
        while fringe:
            current_state = fringe.pop()
            if current_state.is_goal():
                print("visited: ", len(visited))
                return current_state.traceback_path()
            successors = set(
                filter(lambda state: state not in visited, current_state.successors())
            )
            fringe.extend(successors)
            visited.update(successors)

    def sahc(self, heuristic_func):
        """Steepest Accent Hill Climbing with Backtracking"""

        fringe = deque([self])
        visited = set([self])
        while fringe:
            current_state = fringe.pop()
            if current_state.is_goal():
                print("visited: ", len(visited))
                return current_state.traceback_path()
            successors = sorted(
                filter(lambda state: state not in visited, current_state.successors()),
                key=heuristic_func,
                # state with highest heuristic values or objective function
                # is appended to the right of the fringe
            )
            fringe.extend(successors)
            visited.update(successors)

    def find_solution(self, strategy="dfs"):
        """Find the solution"""
        return self.search_strategies[strategy]()
