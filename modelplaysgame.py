import gym
import tflearn as tf
import numpy as np

env = gym.make('CartPole-v0')
print(env.observation_space)
print(env.action_space)
env.reset()
action = env.action_space.sample()
print(action)
observation, reward, done, info = env.step(action)
print(observation)
score_requirement=50
goal_steps = 500
initial_games = 1000
LR = 1e-3  # 1*10^-3 # .001, learning rate


def initial_population():  # training data
    train_data = []
    accepted_scores = []

    print("Playing random Games")

    for _ in range(initial_games):
        env.reset()

        game_memory = []

        prev_observation = []

        score = 0

        # within each game do this

        for x in range(goal_steps):  # x goes from 0 to 499
            action = env.action_space.sample()
            observation, reward, done, info=env.step(action)

            score += reward

            if (x > 0):
                game_memory.append([prev_observation, int(action)])

            prev_observation = observation

            if done:
                break

        if score > score_requirement:
                accepted_scores.append(score)

                for data in game_memory:
                    if data[1] == 1:
                        output = [0, 1] # store 1 as vector instead of right direction
                    elif data[1] == 0:
                        output = [1, 0]
                    train_data.append([data[0], output])

    print(accepted_scores)
    return train_data


def neural_net_model(input_size):
    net = tf.input_data(shape=[None, input_size], name='input')

    # net = tf.input_data(shape=[None, input_size], name='inputlayer')
    net = tf.fully_connected(net, 128, activation='relu')  # network, number of nodes in layer, activation function using linear regression(default)
    net = tf.dropout(net, 0.8)  # dropout occasionally blocks off neurons so work is evenly spread between all neurons
    net = tf.fully_connected(net, 256, activation='relu', name='hiddenlayer2')
    net = tf.dropout(net, 0.8)
    net = tf.fully_connected(net, 512, activation='relu', name='hiddenlayer3')
    net = tf.dropout(net, 0.8)
    net = tf.fully_connected(net, 256, activation='relu', name='hiddenlayer4')
    net = tf.dropout(net, 0.8)
    net = tf.fully_connected(net, 128, activation='relu', name='hiddenlayer5')
    net = tf.dropout(net, 0.8)

    # softmax makes outputs probabilistic, makes add up to 1 (heads, tails)
    net = tf.fully_connected(net, 2, activation='softmax', name='outputlayer1')
    # Error message: ValueError: Cannot feed value of shape (64, 2) for Tensor 'TargetsData/Y:0', which has shape '(?, 9)'
    # 2nd value in net is the value 9 in shape(?, 9)

    # linear regression, line of best fit; LR should be just right to find line of best fit
    net = tf. regression(net, learning_rate=LR)

    model = tf.DNN(net, tensorboard_dir='log')   # make a Deep Neural Net with with network

    return model


def train_model(train_data):
    print("training data is")
    print(train_data)
    X = [i[0] for i in train_data]  # reshape(-1,len(training_data[0][0]),1)
    y = []
    for i in train_data:
        y.append(i[1])

    print("lenx1 is ")
    print(X)
    model = neural_net_model(input_size=len(X[1]))
    model.fit(X, y, n_epoch=5, show_metric=True, run_id='openai_learning')
    return model

    # x = [i[0] for i in train_data]
    # y = []
    # for i in train_data:
    #     y.append(i[1])
    #
    # model = neural_net_model(input_size=len(x[1]))
    # model.fit(x, y, n_epoch=5, show_metric=True, run_id='openai_learning')  # feed training data 5 times
    # return model


def play_with_model(model):
    scores = []
    choices = []
    print("Playing wtih Trained Model.....")

    for each_game in range(10):
        score = 0
        game_memory = []
        prev_observation = []
        env.reset()
        for steps in range(goal_steps):
            env.render()
            if len(prev_observation)==0:
                action = env.action_space.sample()
            else:

                action = np.argmax(model.predict([prev_observation]))

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_observation = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)

    print(scores)


play_with_model(train_model(initial_population()))

