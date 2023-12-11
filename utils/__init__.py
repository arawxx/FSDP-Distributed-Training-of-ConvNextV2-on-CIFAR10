from datetime import datetime


def get_date_of_run():
    ''' create date and time for file saving uniqueness
    example: 2023-08-12-08:31:12_PM'
    '''
    date_of_run = datetime.now().strftime('%Y-%m-%d-%I:%M:%S_%p')
    print(f'--> current date and time of run = {date_of_run}')
    return date_of_run
