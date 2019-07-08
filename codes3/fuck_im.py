from utility import my_execfile, send_notification
counter_auto_restart = 0
try:
    my_execfile('./one_script.py', g=globals(), l=locals())
except Exception as error:
    counter_auto_restart += 1
    send_notification(text = ad.solver.fea_config_dict['pc_name'] + str(error) + '\n'*3,
                      subject = '[%s] Auto Restart Run #%d'%(ad.solver.fea_config_dict['pc_name'], counter_auto_restart))
