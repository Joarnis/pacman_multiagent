# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        "Chose not to add anything here"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        new_pos = successorGameState.getPacmanPosition()
        old_food = currentGameState.getFood()
        new_ghoststates = successorGameState.getGhostStates()
        new_foodlist = successorGameState.getFood().asList()
        new_capsule_list = successorGameState.getCapsules()

        # Stay away from ghosts if they are not scared
        away_from_ghosts = 0
        chase_scared = 0
        for ghost_state in new_ghoststates:
            ghost_dist = manhattanDistance(new_pos, ghost_state.getPosition())
            if ghost_state.scaredTimer == 0:
                if ghost_dist < 2:
                    away_from_ghosts -= 10
            # If a ghost is scared and pacman can reach it, then chase it
            elif ghost_state.scaredTimer > ghost_dist:
                chase_scared += 10.0 / (ghost_state.scaredTimer + ghost_dist)

        # Go towards the closest power capsule, if there are any
        eat_capsules = 10
        if new_capsule_list:
            min_dist = min(manhattanDistance(new_pos, capsule) for capsule in new_capsule_list)
            eat_capsules = 10.0 / min_dist

        eat_food = 1000
        # If there was food in new location (the distance from the nearest food is 0)
        if old_food[new_pos[0]][new_pos[1]]:
            eat_food = 10
        # Go towards the closest food (if there is any) and make a priority to lower the remaining food number
        # (Note that the distance from closest food will never be 0 (successor state))
        elif new_foodlist:
            min_dist = min(manhattanDistance(new_pos, foodcoord) for foodcoord in new_foodlist)
            eat_food = 9.0 / (min_dist + 10 * len(new_foodlist))

        # Prioritise pacman's safety, then the scared ghost chasing, then food eating and being mobile (general score)
        # and lastly, power capsule eating
        return 20 * away_from_ghosts + 5 * chase_scared + 0.1 * eat_capsules + 1 * eat_food \
            + 8 * currentGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # Minimax function
        legal_actions = gameState.getLegalActions(0)
        possible_gameStates = (gameState.generateSuccessor(0, action) for action in legal_actions)
        min_values = [self.min_val(pos_gameState, 0, 1) for pos_gameState in possible_gameStates]
        max_index = min_values.index(max(min_values))
        return legal_actions[max_index]

    def min_val(self, gameState, curr_depth, curr_agent):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legal_actions = gameState.getLegalActions(curr_agent)
        possible_gameStates = (gameState.generateSuccessor(curr_agent, action) for action in legal_actions)
        agent_num = gameState.getNumAgents()
        # If there are more ghosts, then generate all possible min_val nodes
        if curr_agent < agent_num - 1:
            min_values = (self.min_val(pos_gameState, curr_depth, curr_agent + 1) for pos_gameState in possible_gameStates)
            return min(min_values)
        # Else start generating max nodes
        else:
            max_values = (self.max_val(pos_gameState, curr_depth + 1) for pos_gameState in possible_gameStates)
            return min(max_values)

    def max_val(self, gameState, curr_depth):
        if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
            return self.evaluationFunction(gameState)

        legal_actions = gameState.getLegalActions(0)
        possible_gameStates = (gameState.generateSuccessor(0, action) for action in legal_actions)
        min_values = (self.min_val(pos_gameState, curr_depth, 1) for pos_gameState in possible_gameStates)
        return max(min_values)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Alpha-Beta function
        alfa = -float("inf")
        beta = float("inf")

        ni = -float("inf")
        legal_actions = gameState.getLegalActions(0)
        i = 0
        maxi = 0
        for action in legal_actions:
            pos_gameState = gameState.generateSuccessor(0, action)
            min_value = self.min_val(pos_gameState, alfa, beta, 0, 1)
            if ni < min_value:
                ni = min_value
                maxi = i
            if ni > beta:
                return ni
            if alfa < ni:
                alfa = ni
            i += 1
        return legal_actions[maxi]


    def min_val(self, gameState, alfa, beta, curr_depth, curr_agent):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        ni = float("inf")
        legal_actions = gameState.getLegalActions(curr_agent)
        agent_num = gameState.getNumAgents()
        # If there are more ghosts, then generate all possible min_val nodes
        if curr_agent < agent_num - 1:
            for action in legal_actions:
                pos_gameState = gameState.generateSuccessor(curr_agent, action)
                min_value = self.min_val(pos_gameState, alfa, beta, curr_depth, curr_agent + 1)
                if ni > min_value:
                    ni = min_value
                if ni < alfa:
                    return ni
                if beta > ni:
                    beta = ni
            return ni
        # Else start generating max nodes
        else:
            for action in legal_actions:
                pos_gameState = gameState.generateSuccessor(curr_agent, action)
                max_value = self.max_val(pos_gameState, alfa, beta, curr_depth + 1)
                if ni > max_value:
                    ni = max_value
                if ni < alfa:
                    return ni
                if beta > ni:
                    beta = ni
            return ni

    def max_val(self, gameState, alfa, beta, curr_depth):
        if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
            return self.evaluationFunction(gameState)

        ni = -float("inf")
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            pos_gameState = gameState.generateSuccessor(0, action)
            min_value = self.min_val(pos_gameState, alfa, beta, curr_depth, 1)
            if ni < min_value:
                ni = min_value
            if ni > beta:
                return ni
            if alfa < ni:
                alfa = ni
        return ni

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # Expectimax function
        legal_actions = gameState.getLegalActions(0)
        possible_gameStates = (gameState.generateSuccessor(0, action) for action in legal_actions)
        chance_values = [self.chance_val(pos_gameState, 0, 1) for pos_gameState in possible_gameStates]
        max_index = chance_values.index(max(chance_values))
        return legal_actions[max_index]


    def chance_val(self, gameState, curr_depth, curr_agent):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legal_actions = gameState.getLegalActions(curr_agent)
        possible_gameStates = (gameState.generateSuccessor(curr_agent, action) for action in legal_actions)
        agent_num = gameState.getNumAgents()
        chance_values = 0.0
        value_counter = 0.0
        # If there are more ghosts, then generate all possible chance_val nodes
        if curr_agent < agent_num - 1:
            for pos_gameState in possible_gameStates:
                chance_values += self.chance_val(pos_gameState, curr_depth, curr_agent + 1)
                value_counter += 1
            return chance_values / value_counter
        # Else start generating max nodes
        else:
            for pos_gameState in possible_gameStates:
                chance_values += self.max_val(pos_gameState, curr_depth + 1)
                value_counter += 1
            return chance_values / value_counter

    def max_val(self, gameState, curr_depth):
        if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
            return self.evaluationFunction(gameState)

        legal_actions = gameState.getLegalActions(0)
        possible_gameStates = (gameState.generateSuccessor(0, action) for action in legal_actions)
        chance_values = (self.chance_val(pos_gameState, curr_depth, 1) for pos_gameState in possible_gameStates)
        return max(chance_values)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: My betterEvaluationFunction is almost identical to my
        reflex agent evaluationFunction. Firstly, subtract 10 from variable
        away_from_ghosts for each ghost whose distance from pacman is lower
        than 2 and is not scared, so as to prevent pacman from losing the game.
        Next, if a ghost is scared, and pacman can reach it before it switches
        to normal, then add 10.0 divided by the time the ghost will be scared
        plus pacman's distance from the ghost to a variable named chase_scared,
        so as to make pacman chase scared ghosts. Then add 10 divided by pacman's
        distance to the closest power capsule to a variable named eat_capsules.
        Then, is there is any remaining food, divide 10 with pacman's distance to
        the closest food + 10 times the remaining food and store it in variable
        eat_food. This way pacman goes towards the closest food (if there is any)
        and makes it a priority to lower the remaining food number. Lastly combine
        all these variables plus the actual state's score, depending on their importance.
        Prioritise pacman's safety, then the scared ghost chasing, then food eating
        and being mobile (general score) and lastly, power capsule eating.
    """
    # Useful information you can extract from a GameState (pacman.py)
    new_pos = currentGameState.getPacmanPosition()
    new_ghoststates = currentGameState.getGhostStates()
    new_foodlist = currentGameState.getFood().asList()
    new_capsule_list = currentGameState.getCapsules()

    # Stay away from ghosts if they are not scared
    away_from_ghosts = 0
    chase_scared = 0
    for ghost_state in new_ghoststates:
        ghost_dist = manhattanDistance(new_pos, ghost_state.getPosition())
        if ghost_state.scaredTimer == 0:
            if ghost_dist < 2:
                away_from_ghosts -= 10
        # If a ghost is scared and pacman can reach it, then chase it
        elif ghost_state.scaredTimer > ghost_dist:
            chase_scared += 10.0 / (ghost_state.scaredTimer + ghost_dist)

    # Go towards the closest power capsule, if there are any
    eat_capsules = 10
    if new_capsule_list:
        min_dist = min(manhattanDistance(new_pos, capsule) for capsule in new_capsule_list)
        eat_capsules = 10.0 / min_dist

    # Go towards the closest food (if there is any) and make a priority to lower the remaining food number
    # (Note that the distance from closest food will never be 0)
    if new_foodlist:
        min_dist = min(manhattanDistance(new_pos, foodcoord) for foodcoord in new_foodlist)
        eat_food = 10.0 / (min_dist + 10 * len(new_foodlist))
    else:
        eat_food = 1000

    # Prioritise pacman's safety, then the scared ghost chasing, then food eating and being mobile (general score) and
    # lastly, power capsule eating
    return 20 * away_from_ghosts + 5 * chase_scared + 0.1 * eat_capsules + 1 * eat_food \
        + 8 * currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

