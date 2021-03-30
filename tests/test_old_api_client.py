from useis.clients.old_api_client.api_client import events_list

response, seismic_events = events_list(evaluation_mode='manual',
                                       evaluation_status='accepted',
                                       event_type='seismic event')

response, other_events = events_list(evaluation_mode='manual',
                                     evaluation_status='rejected')