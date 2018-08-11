from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])

  for i in range(S):
    alpha[i][0] = pi[i]*B[i][O[0]]

  for i in range(1, N):
    for j in range(S):
      sigma = np.dot(alpha[:,i-1], A[:,j])
      alpha[j][i] = B[j][O[i]]*sigma

  return alpha


def backward(pi, A, B, O):
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])

  for i in range(S):
    beta[i][-1] = 1

  for i in range(N-2,-1, -1):
    for j in range(S):
      sigma = np.dot(np.multiply(B[:, O[i+1]].T, beta[:,i+1]), A[j, :])

      beta[j][i] = sigma

  return beta

def seqprob_forward(alpha):
  prob = 0

  for i in range(len(alpha)):
    prob += alpha[i][-1]

  return prob


def seqprob_backward(beta, pi, B, O):
  prob = 0

  for i in range(len(beta)):
    prob += beta[i][0]*pi[i]*B[i][O[0]]

  return prob

def viterbi(pi, A, B, O):
  path = []

  S = len(pi)
  N = len(O)
  table_d = np.zeros((S, N))
  table_p = np.zeros((S, N))
  path = np.zeros(N)

  for i in range(S):
    table_d[i][0] = pi[i]*B[i][O[i]]

  for i in range(1, N):
    for j in range(S):
      temp = np.multiply(np.multiply(table_d[:, i-1], B[:, O[i]]), A[:, j]);
      table_d[j][i] = np.amax(temp)
      table_p[j][i] = np.argmax(temp)

  path[-1] = np.argmax(table_d[:, -1])

  for i in range(N-1, 0, -1):
    path[i-1] = table_p[int(path[i])][i]

  return path.astype(int).tolist()


def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  obs_symbols = data['observations']
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()