import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

class Sample:
    def user_usage_score(show=False):
        # Usage: Score determining how often a user uses the application 
        shape, scale = 1, 0.1  #Average usage score: 0.1
        usage = np.random.gamma(shape, scale, 1000)

        if show:
            count, bins, ignored = plt.hist(usage, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Usage distribution')
            plt.xlabel('Usage score')
            plt.show()

        return usage[0]

    def usage_thresh():
        mu, sigma = 0.5, 0.1  # Perfectly normal distribution
        res = np.random.normal(mu, sigma, 1)
        return res[0]

    def user_subscription_score(show=False):
        shape, scale = 1, 0.1  #Average usage score: 0.1
        subscib = np.random.gamma(shape, scale, 1000)
        
        if show:
            count, bins, ignored = plt.hist(subscib, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Subscription distribution')
            plt.xlabel('Usage score')
            plt.show()

        return subscib[0]

    def user_indexing_captioning(show=False):
        # Indexing: length of video indexed on a given day by a given user.
        shape, scale = 2, 2  # Average indexing length: 4 min
        indexing = np.random.gamma(shape, scale, 1000)

        if show:
            count, bins, ignored = plt.hist(indexing, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Indexing and Captioning distribution')
            plt.xlabel('Min indexed')
            plt.show()

        return indexing[0]

    def user_generation(show=False):
        # Generation: Length of video generated on a given day by a given user in seconds
        shape, scale = 4, 1  # Average generated length: 4 sec
        generation = np.random.gamma(shape, scale, 1000)

        if show:
            count, bins, ignored = plt.hist(generation, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Generation distribution')
            plt.xlabel('Min indexed')
            plt.show()

        return generation[0]

    def user_storage(show=False):
        # Storage: amount of media uploaded on a given day by a given user (in GB)
        shape, scale = 3, 1  # Average data stored: 3 Gb per use
        storage = np.random.gamma(shape, scale, 1000)

        if show:
            count, bins, ignored = plt.hist(storage, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Storage distribution')
            plt.xlabel('Gb stored')
            plt.show()

        return storage[0]

    def user_search(show=False):
        # Searching: number of searches made on a given day by a given user
        mu, sigma = 15, 0.1  # Average of 15 searches per use
        search = np.random.normal(mu, sigma, 1000)

        if show:
            count, bins, ignored = plt.hist(search, 30, density=True)
            plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
            plt.title('Search distribution')
            plt.xlabel('Number of searches')
            plt.show()

        return search[0]


if __name__ == '__main__':
    Sample.user_usage_score(show=True) 
    Sample.user_subscription_score(show=True)
    Sample.user_indexing_captioning(show=True)
    Sample.user_generation(show=True)
    Sample.user_storage(show=True)
    Sample.user_search(show=True)
