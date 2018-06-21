# Predicting wind speed using time series techniques

## Project overview
More and more of the world's power is being generated from renewable energy sources such as solar and wind that inherently produce variable output, due to their dependence on weather conditions. Grid operators need to balance supply and demand across different sources of energy, and so producing accurate forecasts for renewable power generation is important so as not to over or under-order power from other sources. Under-ordering could lead to blackouts whereas over-ordering can lead to fires from power surges as well as unnecessary emmissions. Considering wind energy specifically, as power output from a wind farm largely depends on the wind speed, having accurate forecasts for wind dynamics will allow for better predictions in power output and thus improve grid efficiency. 

The goal of this project is to use time series analysis to create an accurate hourly forecast for wind speed at 100m altitude over one day across the state of Wyoming. Wyoming was selected as there are large variations in wind speeds across the state, so the final model will be valid for a full range of wind speeds. As an added bonus, Wyoming is square in shape, making it easier to extract Wyoming specific data from the US-wide dataset. I also have a personal soft-spot for Wyoming having spent part of my engagement honeymoon in the incredible Yellowstone National Park!

Model performance will be guided by the RMSE from true wind speed values. The model will initially be built using a pure ARIMA model on historical wind speeds, and then enhanced by incorporating other relevant factors such as temperature and pressure. 

## Data source

The dataset used for modeling is the <a href="https://www.nrel.gov/grid/wind-toolkit.html">NREL WIND toolkit</a>. A newly enhanced public version of the dataset was released in May 2018, and contains 50TB of data covering barometric pressure, wind speed and direction, relative humidity, temperature, and air density data from 2007 to 2013, in hourly intervals, on a uniform 2km grid that covers the continental US and beyond. (The previous version released in 2015 contained 2TB of data.)

