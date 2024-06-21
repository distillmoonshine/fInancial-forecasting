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
        self.new_users_by_day = None
        self.users_by_day = []
        self.user_count = None

        self.user_id = 0
        self.user_ids = []
        self.user_data = None

        self.indexed_by_day = []
        self.stored_by_day = []
        self.searched_by_day = []

        self.ops_cost_by_day = []
        self.ops_cost = 0

        self.r_and_d_cost_by_day = []
        self.r_and_d_cost = 0

        self.talent_cost_by_day = []
        self.talent_cost = 0
        
        self.cost_by_day = []
        self.total_cost = 0

        self.revenue_by_day = []
        self.total_revenue = 0

        self.profit_by_day = []
        self.total_profit = 0


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
        
        for count,user in enumerate(users):
            self.users_by_day.append(np.sum(users[0:count]))

        if show:
            plt.plot(days, users, label='New users per day')
            plt.plot(days, self.users_by_day, label='Total users per day')
            plt.title('User aquisiton, total users: ' + str(int(total_users)))
            plt.legend(loc='best')
            plt.xlabel('Days')
            plt.ylabel('Users')
            plt.show()
        
        self.days = days
        self.new_users_by_day = users
        self.user_count = total_users
        return self.days, self.new_users_by_day, self.users_by_day, self.user_count

    def sample_user_usage_distribution(self, size, show=False):
        # usage: Score determining how often a user uses the application 
        shape, scale = 1, 0.1  #Average usage score: 0.1
        usage = np.random.gamma(shape, scale, size)

        if show:
            count, bins, ignored = plt.hist(usage, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Usage distribution')
            plt.xlabel('Usage score')
            plt.show()

        return usage

    def sample_usage_thresh(self):
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
    
    def sample_user_storage_distribution(self, size, show=False):      
        # storage: amount of media uploaded on a given day by a given user (in GB)
        shape, scale = 3, 1  # Average data stored: 3 Gb per use
        storage = np.random.gamma(shape, scale, size)

        if show:
            count, bins, ignored = plt.hist(storage, 50, density=True)
            y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
            plt.plot(bins, y, linewidth=2, color='r')  
            plt.title('Storage distribution')
            plt.xlabel('Gb stored')
            plt.show()

        return storage

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
        mb_uploaded = self.sample_user_storage_distribution(1)[0]
        searches_made = self.sample_user_search_distribution(1)[0]
        user = np.array([user_id, usage_score, min_indexed, mb_uploaded, searches_made])
        self.user_data = np.vstack((self.user_data, user))  # Add user ID to array
        self.user_ids.append(user_id)  # Append user ID Links
        return user

    def predict_usage(self):
        step_usage_thresh = self.sample_usage_thresh()
        for user_id in self.user_ids:
            if self.user_data[user_id, 1] >= step_usage_thresh:
                self.user_data[user_id, 2] += self.sample_user_index_distribution(1)[0]
                self.user_data[user_id, 3] += self.sample_user_storage_distribution(1)[0]
                self.user_data[user_id, 4] += self.sample_user_search_distribution(1)[0]

    def calculate_usage(self, show=True):
        self.user_data = np.empty((0, 5))
        for count, day in enumerate(self.days):
            new_users = self.new_users_by_day[count]
            for user in range(int(new_users)):
                self.create_user()
            self.predict_usage()
            self.indexed_by_day.append(np.sum(self.user_data[:,2]))
            self.stored_by_day.append(np.sum(self.user_data[:,3]))
            self.searched_by_day.append(np.sum(self.user_data[:,4]))

        if show:
            plt.plot(self.days, self.indexed_by_day, label='indexing use')
            plt.plot(self.days, self.stored_by_day, label='storage use')
            plt.plot(self.days, self.searched_by_day, label='search use')
            plt.legend(loc='best')
            plt.title('Application use over time')
            plt.xlabel('Days')
            plt.show()

        return self.days, self.indexed_by_day, self.stored_by_day, self.searched_by_day

    def calculate_ops_cost(self, show=True):
        # Vercel: neglibible
        cost_per_min_indexed = 0.04  #Includes captioning 
        cost_per_gb_uploaded = 0.023/30
        cost_per_search = 0.0017 
        base_cost_per_day = 15.6 
        
        for count, day in enumerate(self.days):
            cost = 0
            cost += self.indexed_by_day[count]*cost_per_min_indexed
            cost += self.stored_by_day[count]*cost_per_gb_uploaded
            cost += self.searched_by_day[count]*cost_per_search
            cost += base_cost_per_day
            self.ops_cost_by_day.append(cost)
        total_cost = sum(self.ops_cost_by_day)

        cost_over_time = [self.ops_cost_by_day[0]]
        for count in range(len(self.ops_cost_by_day)-1):
            cost_over_time.append(cost_over_time[count-1] + cost_over_time[count])

        if show:
            plt.plot(self.days, self.ops_cost_by_day, label='Daily cost')
            plt.plot(self.days, cost_over_time, label='Cost over time')
            plt.legend(loc='best')
            plt.title('Operations cost: '+ str(int(total_cost)))
            plt.xlabel('Days')
            plt.ylabel('Dollars')
            plt.show()
        
        self.ops_cost = total_cost
        return self.ops_cost, self.ops_cost_by_day

    def calculate_r_and_d_cost(self, show=True):
        init_cost = 1784.93
        k = 1/57 #TODO
        total_cost = []
        cost_by_day = [init_cost]
        for count, day in enumerate(self.days):
            total_cost.append(init_cost*np.e**(k*count))
        for count in range(len(total_cost)-1):
            cost_by_day.append(total_cost[count+1] - total_cost[count])
        self.r_and_d_cost_by_day = cost_by_day
        self.r_and_d_cost = total_cost[-1]

        if show:
            plt.plot(self.days, cost_by_day, label='Daily cost')
            plt.plot(self.days, total_cost, label='Cost over time')
            plt.title('R&D cost, total: ' + str(int(self.r_and_d_cost)))
            plt.xlabel('Days')
            plt.ylabel('Dollars')
            plt.legend(loc='best')
            plt.show()

        return self.r_and_d_cost, self.r_and_d_cost_by_day

    def calculate_talent_cost(self, show=True):
        # Annual salaries:
        salaries = [75000, 75000, 100000, 150000, 100000] #TODO
        
        daily_salary = []
        for salary in salaries:
            daily_salary.append(salary/365)
        daily_salary_sum = sum(daily_salary)
        
        salary_by_day = []
        total_spend_salary_by_day = []
        for day in self.days:
            salary_by_day.append(daily_salary_sum)
            total_spend_salary_by_day.append(daily_salary_sum*day)
        self.talent_cost = total_spend_salary_by_day[-1]

        if show:
            plt.plot(self.days, salary_by_day, label='Daily cost')
            plt.plot(self.days, total_spend_salary_by_day, label='Cost over time')
            plt.title('Cost of talent, total cost: ' + str(int(self.talent_cost)))
            plt.legend(loc='best')
            plt.xlabel('Day')
            plt.ylabel('Cost')
            plt.show()

        self.talent_cost_by_day = salary_by_day
        return self.talent_cost, self.talent_cost_by_day
    
    def calculate_total_cost(self, show=True):
        for count, day in enumerate(self.days):
            cost = 0
            cost += self.ops_cost_by_day[count]
            cost += self.r_and_d_cost_by_day[count]
            cost += self.talent_cost_by_day[count]
            self.cost_by_day.append(cost)
        self.total_cost = sum(self.cost_by_day)

        if show:
            plt.plot(self.days, self.cost_by_day, label='Daily cost')
            plt.title('Total cost: ' + str(int(self.total_cost)))
            plt.xlabel('Days')
            plt.ylabel('Dollars')
            plt.legend(loc='best')
            plt.show()

        return self.total_cost, self.cost_by_day

    def calculate_revenue_profit(self, show=True):
        user_cost_per_min_indexed = 0.06  #TODO -> 10 hours for 36 dollars
        user_cost_per_mb_stored = 0  #TODO
        user_cost_per_search = 0  #TODO

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
            plt.plot(self.days, self.revenue_by_day, label='Revenue by day')
            plt.plot(self.days, self.profit_by_day, label='Profit by day')
            plt.legend(loc='best')
            plt.title('Total revenue: '+str(int(total_revenue))+', Total profit: '+str(int(total_profit)))
            plt.xlabel('Days')
            plt.ylabel('Dollars')
            plt.show()
        
        self.total_revenue = total_revenue
        self.total_profit = total_profit
        return self.total_revenue, self.total_profit

    def pie_chart_of_costs(self, show=True):
        labels = 'Operations', 'Research', 'Talent'
        costs = [self.ops_cost, self.r_and_d_cost, self.talent_cost]

        if show:
            plt.pie(costs, labels=labels, autopct='%1.1f%%')
            plt.title('Breakdown of costs, total: ' + str(int(sum(costs))))
            plt.show()
        
        return sum(costs)

    def pie_chart_total_budget(self, budget, show=True):
        labels = 'Operations', 'Research', 'Talent', 'Misc'
        costs = [self.ops_cost, self.r_and_d_cost, self.talent_cost]
        total_spend = sum(costs)

        misc_spend = budget - total_spend
        costs.append(misc_spend)
        
        if show:
            plt.pie(costs, labels=labels, autopct='%1.1f%%')
            plt.title('Breakdown of coss, total: ' + str(int(sum(costs))))
            plt.show()
        
        return sum(costs)

    
if __name__ == '__main__':
    # Given: Start date, End Date [[Date, predicted signups], [], ...]
    start_date = (2024, 6, 17)  # not included in analysis
    
    end_date = (2025, 6, 1)
    user_aquisition_dates = [[(2024, 6, 18), 100], [(2024, 7, 12), 1000], [(2024, 8, 15), 1500], [(2024, 9, 15), 2000]]
    
    #end_date = (2024, 7, 17)
    #user_aquisition_dates = [[(2024, 6, 18), 2000]]


    model = Model(
        start_date=start_date,
        end_date=end_date,
        user_aquisition_dates=user_aquisition_dates
    )
    
    # Create the user aquistion curve
    model.generate_user_aquisition_curve()

    # View distributions for usage, indexing, storage, and searches
    model.sample_user_usage_distribution(1000, show=True)
    model.sample_user_index_distribution(1000, show=True)
    model.sample_user_storage_distribution(1000, show=True)
    model.sample_user_search_distribution(1000, show=True)

    # Create a plot of usage over time for indexing, storage, and searching
    model.calculate_usage()

    # Create a plot of cost, revenue, and profit over time
    model.calculate_ops_cost()

    # Create a plot of R and D costs (GPU up time and use) over time
    model.calculate_r_and_d_cost()

    # Create a plot of talent costs over time
    model.calculate_talent_cost()

    # Create a chart of total cost over time
    model.calculate_total_cost()

    # Create a plot of revenue and profit over time    
    model.calculate_revenue_profit()

    # Create a pie chart of costs from start date to end date
    model.pie_chart_of_costs()

    # Create a pie chart of costs to a certain date within the start and end date
    model.pie_chart_total_budget(3000000)  #TODO


