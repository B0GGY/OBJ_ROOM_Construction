import numpy as np
from matplotlib import pyplot as plt

def get_metrics(n, h):
    n = np.array(n)
    h = np.array(h)
    data_value = []
    for i in range(6):
        data_value.append(max(n[h==i]))
    overall_score = 0
    for i in range(6):
        for j in range(i+1, 6):
            gradient_score = get_gradient_score(j, i, data_value)
            overall_score += (1/1)*gradient_score#(1/sum(h==i))*gradient_score
            # print(j, i, gradient_score)
            print(j, i, (1/1)*gradient_score)
        print('----')
    print(overall_score)
    return overall_score

def get_gradient_score(i, j, values):

    gradient = (values[i]-values[j])/(i-j)
    # print(gradient, 1-np.exp(gradient))
    return 1-np.exp(gradient)

if __name__ == '__main__':
    room_list = ['Bathroom', 'Bedroom', 'Dining Room', 'gym', 'Kitchen', 'Laundry', 'Living Room', 'Study']
    # room_list = ['Bathroom']
    for room in room_list:
        n = np.load('test_output_generalizability/{}/n.npy'.format(room))
        h = np.load('test_output_generalizability/{}/h.npy'.format(room))
        overall_score = get_metrics(n, h / 2)
        plt.scatter(h/2, n)
        plt.xlabel("Number of replaced items", fontsize=14)
        plt.ylabel("Similarity",fontsize=14)
        plt.title('{} {}'.format(room,overall_score))
        plt.show()
