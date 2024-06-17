import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sps
import datetime


class Model:
    def __init__(self, start_date, end_date, user_aquisition_dates):
        self.start_date = datetime.date(start_date[0], start_date[1], start_date[2])
        self.end_date = datetime.date(end_date[0], end_date[1], end_date[2])
        self.user_aquisition_dates = []
        self.user_aquisition_counts = []
        for date in user_aquisition_dates:
            self.user_aquisition_dates.append(datetime.date(date[0][0], date[0][1], date[0][2]))
            self.user_aquisition_counts.append(date[1])
        
        self.days = None
        self.users_by_day = None
        self.user_count = None

        self.user_id = 0
        self.user_ids = []
        self.user_data = None

        self.indexed_by_day = []
        self.stored_by_day = []
        self.searched_by_day = []

        self.cost_by_day = []
        self.revenue_by_day = []
        self.profit_by_day = []

    def generate_user_aquisition_curve(self, show=True):
        delta = self.end_date - self.start_date  
        days = np.linspace(1, delta.days, delta.days)
        users = np.ones(days.shape)*-1
        for count, date in enumerate(self.user_aquisition_dates):
            delta = date - self.start_date
            users[delta.days-1] = self.user_aquisition_counts[count]
        
        # Apply exponential decay between given dates
        k = 1/(1.6)  # Decay rate (Tuned to linkedIn data)
        y_0 = 0
        day_0 = 0        
        for count, user_count in enumerate(users):
            if user_count != -1:
                y_0 = user_count
                day_0 = days[count]
            if user_count == -1:
                users[count] = y_0*np.e**(-k*(days[count] - day_0))
        users = np.round(users, decimals=0)
        total_users = np.sum(users)

        if show:
            plt.plot(days, users)
            plt.title('User aquisiton curve, total users: ' + str(int(total_users)))
            plt.show()
        
        self.days = days
        self.users_by_day = users
        self.user_count = total_users
        return self.days, self.users_by_day, self.user_count

    def sample_user_usage_distribution(self, size, show=False):
        # usage: Score determining how often a user uses the application 
        shape, scale = 1, 0.1  #Average useage score: 0.1
        useage = np.random.gamma(shape, scale, size)

        if show:
            count, bins, ignored = plt.hist(useage, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Useage distribution')
            plt.xlabel('Useage score')
            plt.show()

        return useage

    def sample_useage_thresh(self):
        mu, sigma = 0.5, 0.1  # Perfectly normal distribution
        res = np.random.normal(mu, sigma, 1)
        return res[0]

    def sample_user_index_distribution(self, size, show=False):
        # indexing: length of video indexed on a given day by a given user.
        shape, scale = 2, 2  # Average indexing length: 4 min
        indexing = np.random.gamma(shape, scale, size)

        if show:
            count, bins, ignored = plt.hist(indexing, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Indexing distribution')
            plt.xlabel('Min indexed')
            plt.show()

        return indexing
    
    def sample_user_storeage_distribution(self, size, show=False):      
        # storeage: amount of media uploaded on a given day by a given user (in GB)
        shape, scale = 3, 1  # Average data stored: 5 Gb per use
        storeage = np.random.gamma(shape, scale, size)

        if show:
            count, bins, ignored = plt.hist(storeage, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Storeage distribution')
            plt.xlabel('Gb stored')
            plt.show()

        return storeage

    def sample_user_search_distribution(self, size, show=False):
        # searching: number of searches made on a given day by a given user
        mu, sigma = 15, 0.1  # Average of 15 searches per use
        search = np.random.normal(mu, sigma, size)

        if show:
            count, bins, ignored = plt.hist(search, 30, density=True)
            plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
            plt.title('Search distribution')
            plt.xlabel('Number of searches')
            plt.show()

        return search

    def create_user(self):
        user_id = self.user_id
        self.user_id += 1        
        usage_score = self.sample_user_usage_distribution(1)[0]
        min_indexed = self.sample_user_index_distribution(1)[0]
        mb_uploaded = self.sample_user_storeage_distribution(1)[0]
        searches_made = self.sample_user_search_distribution(1)[0]
        user = np.array([user_id, usage_score, min_indexed, mb_uploaded, searches_made])
        self.user_data = np.vstack((self.user_data, user))  # Add user ID to array
        self.user_ids.append(user_id)  # Append user ID Links

    def predict_usage(self):
        step_useage_thresh = self.sample_useage_thresh()
        for user_id in self.user_ids:
            if self.user_data[user_id, 1] >= step_useage_thresh:
                self.user_data[user_id, 2] += self.sample_user_index_distribution(1)[0]
                self.user_data[user_id, 3] += self.sample_user_storeage_distribution(1)[0]
                self.user_data[user_id, 4] += self.sample_user_search_distribution(1)[0]

    def calculate_useage(self, show=True):
        self.user_data = np.empty((0, 5))
        for count, day in enumerate(self.days):
            new_users = self.users_by_day[count]
            for user in range(int(new_users)):
                self.create_user()
            self.predict_usage()
            self.indexed_by_day.append(np.sum(self.user_data[:,2]))
            self.stored_by_day.append(np.sum(self.user_data[:,3]))
            self.searched_by_day.append(np.sum(self.user_data[:,4]))

        if show:
            plt.plot(self.days, self.indexed_by_day, label='indexing use')
            plt.plot(self.days, self.stored_by_day, label='storeage use')
            plt.plot(self.days, self.searched_by_day, label='search use')
            plt.legend(loc='best')
            plt.title('Application use over time')
            plt.show()

        return self.days, self.indexed_by_day, self.stored_by_day, self.searched_by_day

    def calculate_cost_revenue_profit(self, show=True):
        # Vercel: neglibible
        cost_per_min_indexed = 0.03 
        cost_per_gb_uploaded = 0.023/30
        cost_per_search = 0.0017 
        base_cost_per_day = 15.6 

        user_cost_per_min_indexed = 0.06  #TODO
        user_cost_per_mb_stored = 0  #TODO
        user_cost_per_search = 0  #TODO
        
        for count, day in enumerate(self.days):
            cost = 0
            cost += self.indexed_by_day[count]*cost_per_min_indexed
            cost += self.stored_by_day[count]*cost_per_gb_uploaded
            cost += self.searched_by_day[count]*cost_per_search
            cost += base_cost_per_day
            self.cost_by_day.append(cost)
        total_cost = sum(self.cost_by_day)

        for count, day in enumerate(self.days):
            revenue = 0
            revenue += self.indexed_by_day[count]*user_cost_per_min_indexed
            self.revenue_by_day.append(revenue)
        total_revenue = sum(self.revenue_by_day)

        for count, day in enumerate(self.days):
            profit = 0
            profit += self.revenue_by_day[count]-self.cost_by_day[count]
            self.profit_by_day.append(profit)
        total_profit = sum(self.profit_by_day)

        if show:
            plt.plot(self.days, self.cost_by_day, label='cost')
            plt.plot(self.days, self.revenue_by_day, label='revenue')
            plt.plot(self.days, self.profit_by_day, label='profit')
            plt.legend(loc='best')
            plt.title('Total cost: '+str(int(total_cost))+', Total revenue: '+str(int(total_revenue))+', Total profit: '+str(int(total_profit)))
            plt.xlabel('Days')
            plt.ylabel('')
            plt.show()


if __name__ == '__main__':
    # Given: Start date, End Date [[Date, predicted signups], [], ...]
    start_date = (2024, 6, 17)  # not included in analysis
    end_date = (2024, 7, 31)
    user_aquisition_dates = [[(2024, 6, 21), 100], [(2024, 6, 25), 400]]

    model = Model(
        start_date=start_date,
        end_date=end_date,
        user_aquisition_dates=user_aquisition_dates
    )
    
    # Create the user aquistion curve
    model.generate_user_aquisition_curve()

    # View distributions for useage, indexing, storage, and searches
    #model.sample_user_usage_distribution(1000, show=True)
    #model.sample_user_index_distribution(1000, show=True)
    #model.sample_user_storeage_distribution(1000, show=True)
    #model.sample_user_search_distribution(1000, show=True)

    # Create a plot of useage over time for indexing, storage, and searching
    model.calculate_useage()

    # Create a plot of cost, revenue, and profit over time
    model.calculate_cost_revenue_profit()



