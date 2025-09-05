# Constants/Variables used in the project
MAJOR_HUBS = ['ATL','DXB','DFW','HND','LHR','DEN','ORD','IST','PVG','ICN','CDG', 'JFK','CLT','MEX','SFO','EWR','MIA','BKK','GRU','HKG']
CUSTOMER_FEATURES = ['companyID', 'sex', 'nationality', 'isVip', 'bySelf']
UNNEEDED_FEATURES = [
    'frequentFlyer', 'frequent_flyer', 'isAccess3D', 'requestDate', 'searchRoute', 'totalPrice', 'taxes', 'legs0_arrivalAt', 'legs0_duration'
]
UNNEEDED_FEATURES_REGEX = r'^legs[01].*$|^leg0_duration_q.*$|^price_q.*$'
POLARS_INDEX_COL = ['__index_level_0__']
