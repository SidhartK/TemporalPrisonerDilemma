import numpy as np 


class Environment():
	def __init__ (self, oppo_prob=0.5, oppo_stdev = 0.01, oppo_updateSize=0.05, startMoney=100, startYears=20):
		self.startMoney = startMoney
		self.startYears = startYears
		self.your_money = startMoney
		self.your_last_action = 1
		self.observed_oppo_prob = 0
		self.oppo_LastMoves = np.full((5), -1.0)

		self.start_oppo_prob = oppo_prob
		self.oppo_prob = oppo_prob
		self.oppo_stdev = oppo_stdev
		self.oppo_updateSize = oppo_updateSize
		self.oppo_money = startMoney

		self.num_rounds = 0
		self.num_years = startYears
		self.num_actions = []
		self.oppo_actions = []
		self.oppo_probs = []
		self.actual_score_history = []
		self.actual_score = 0


	def smooth_transition(self, x):
		if x >= 1:
			return 1.0
		if x <= 0:
			return 0.0
		if x < 0.5:
			return (np.exp(-1/x)/(np.exp(-1/x) + np.exp(-1/(0.5-x)))) * 0.5
		if x > 0.5:
			return (np.exp(-1/(x - 0.5))/(np.exp(-1/(x - 0.5)) + np.exp(-1/(1-x)))) * 0.5 + 0.5
		return 0.5

	def rough_transition(self, x):
		if x > 1:
			return 1
		if x < 0: 
			return 0
		return x

	def update_oppo(self, moreCoop):
		if (moreCoop): 
			self.oppo_prob = self.rough_transition(self.oppo_prob + self.oppo_updateSize)
		else:
			self.oppo_prob = self.rough_transition(self.oppo_prob - self.oppo_updateSize)

	def step(self, action):
		oppo_coopProb = np.random.normal(self.oppo_prob, self.oppo_stdev)
		oppo_action = np.random.random() < oppo_coopProb
		self.oppo_probs.append(oppo_coopProb)
		# oppo_action = self.your_last_action

		self.oppo_actions.append(oppo_action)
		reward = 0
		done = False
		if action and oppo_action: 
			reward = 2
			self.your_money += 2
			self.oppo_money += 2
			self.num_years -= 1
		if action and not oppo_action: 
			reward = -4
			self.your_money += -4
			self.oppo_money += 1
			self.num_years -= 2
		if not action and oppo_action:
			reward = 1
			self.your_money += 1
			self.oppo_money += -4
			self.num_years -= 2
		if not action and not oppo_action:
			reward = -2
			self.your_money += -2
			self.oppo_money += -2
			self.num_years -= 2

		self.observed_oppo_prob = ((self.observed_oppo_prob * self.num_rounds) + oppo_action)/(self.num_rounds + 1)
		self.oppo_LastMoves = self.oppo_LastMoves[1:]
		self.oppo_LastMoves = np.append(self.oppo_LastMoves, [action])
		observation_ = np.append(self.oppo_LastMoves, [self.observed_oppo_prob, self.your_money, self.oppo_money])
		self.update_oppo(action)

		if (self.your_money > 1.1 * self.oppo_money or self.oppo_money < 0):
			reward += 50
			self.your_money += self.oppo_money
			done = True
		elif (self.oppo_money > 1.1 * self.your_money or self.your_money < 0):
			reward += -50
			done = True
		if (self.num_years <= 0):
			if (self.your_money < 0.9 * self.startMoney):
				reward -= 100
			done = True
		self.actual_score += reward
		if done:
			self.actual_score_history.append(self.actual_score)
		reward *= (self.your_money/self.oppo_money)
		self.num_rounds += 1
		self.your_last_action = action
		# if (self.num_rounds >= 10):
		# 	done = True

		# print("********************")
		# print(self.your_money, self.oppo_money, self.num_years, self.num_rounds)
		# print("My action: ", action)
		# print("Opponents features: ", self.oppo_prob, oppo_coopProb, self.oppo_stdev, "Opponent action: ", oppo_action)
		# print(self.oppo_LastMoves)
		# print("Observed opponent probability ", self.observed_oppo_prob)
		if done:
			self.num_actions.append(self.num_rounds)
		return observation_, reward, done

	def reset(self):
		self.oppo_prob = self.start_oppo_prob
		self.observed_oppo_prob = 0
		self.num_rounds = 0
		self.num_years = self.startYears
		self.your_money = self.startMoney
		self.oppo_money = self.startMoney
		self.oppo_LastMoves = np.full((5), -1.0)
		self.actual_score = 0
		observation = np.append(self.oppo_LastMoves, [self.observed_oppo_prob, self.your_money, self.oppo_money])
		return observation


if __name__ == '__main__':
	env = Environment(oppo_stdev=0.02)
	print(env.smooth_transition(2.0), env.smooth_transition(1.0), env.smooth_transition(0.9), env.smooth_transition(0.75), "\n",
			env.smooth_transition(0.55), env.smooth_transition(0.5), env.smooth_transition(0.45), env.smooth_transition(0.25), "\n",
			env.smooth_transition(0.2), env.smooth_transition(0.0), env.smooth_transition(-1.0), "\n")
	done = False
	while not done:
		observation_, reward, done = env.step(np.random.choice([0,1]))
		print("Returned ", observation_.tolist(), reward, done)





