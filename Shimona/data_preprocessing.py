import pandas as pd
from collections import Counter

def available_times(series):
    
    
    index_ = series.index
    
    output = []
    for place in series: #loop through each place
        availability = {} #to store most common availability times of the day and number of days
        if place != None:
            #place = json.loads(place.replace('\'','\"'))
            availability['open_days'] = len(place) #number of days restraunt is open
            meal = [] #count the meals breakfast, lunch or dinner on a particular day
            for week_day in place: 
                
                if (week_day[1][0][0] == 'Open 24 hours') | (week_day[1][0][0] == 'Open'):
                    meal.extend(['Breakfast', 'Lunch', 'Dinner'])
                elif week_day[1][0][0] == 'Closed':
                    #meal.append('Closed')
                    availability['open_days'] -= 1
                
                else:
                    time = week_day[1][0][0].split('--')
                    time = pd.Series(time)
                    try:
                        time = pd.to_datetime(time).dt.hour
                        if time[0]<= 10:
                            meal.append('Breakfast')
                        
                        if (time[0]>=10) | (time[1] <=19):
                            meal.append('Lunch')
                            
                        if (time[0]>=17) | (time[1]<=3) | (time[1] <= 23):
                            meal.append('Dinner')
                    except:
                        print('No timing found for place ID: ',place)
                        
                
                    
            #once we have checked operating times for all days
            #check the maximum of all options and assign restraunt operational time
            #if any two or more options have same value then restraunt is assigned 2 or more
            #operational times
            temp_dict = Counter(meal)
            temp_list = [key for key, value in temp_dict.items() if value == max(temp_dict.values())]
            for item in temp_list:
                availability[item] = 1
        output.append(availability)
    
    output = pd.Series(output)
    output.index = index_

    #ouput is a series with index as Place ID (gPlusPlaceID)
    return output
    
        
