import torch
from torch.distributions import multinomial
from d2l import torch as d2l

if __name__ == '__main__':

    print('--------------------------------------Question 2.6.1--------------------------------------------')
    fair_probs = torch.ones([6]) / 6
    def experiment(n=10,m=500,legend=1):
        counts = multinomial.Multinomial(n, fair_probs).sample((m, ))
        cum_counts = counts.cumsum(dim=0)
        estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)

        d2l.set_figsize((6, 4.5))
        for i in range(6):
            d2l.plt.plot(estimates[:,i].numpy(), label=("P(die=" + str(i + 1) + ")"))

        d2l.plt.title(f'n={n},m={m}')
        d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
        d2l.plt.gca().set_xlabel('Groups of experiments')
        d2l.plt.gca().set_ylabel('Estimated Probability')
        if legend == 1:
            d2l.plt.legend()


    d2l.plt.figure(figsize=(10, 10))
    d2l.plt.subplot(231)
    experiment()
    d2l.plt.subplot(232)
    experiment(1, 500, 0)
    d2l.plt.subplot(233)
    experiment(100, 500, 0)

    d2l.plt.subplot(234)
    experiment(10, 50, 0)
    d2l.plt.subplot(235)
    experiment(1, 50, 0)
    d2l.plt.subplot(236)
    experiment(100, 50, 0)
    d2l.plt.subplots_adjust(wspace=0.5, hspace=0.5)
    d2l.plt.show()

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.6.2--------------------------------------------')
    # 0⩽ P(A∩B)⩽ min(P(A),P(B))
    # max(P(A),P(B))⩽ P(A∪B)⩽ P(A)+P(B)
    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.6.3--------------------------------------------')
    # P(B∣A,C)=P(B∣A)
    # P(C∣A,B)=P(C∣B)

    # P(A, B, C)
    # =P(C∣A, B)P(A, B)
    # =P(C∣A, B)P(B∣A)P(A)
    # =P(C∣B)P(B∣A)P(A)

    # P(A,B,C)=P(C∣B)P(B∣A)P(A)
    print('------------------------------------------------------------------------------------------------\n\n')