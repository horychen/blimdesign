from utility import my_execfile, send_notification
counter_auto_restart = 0
while True:
    try:
        my_execfile('./one_script_pm_2O.py', g=globals(), l=locals())
    except KeyboardInterrupt as error:
        break
    except Exception as error:
        counter_auto_restart += 1
        send_notification(text = ad.solver.fea_config_dict['pc_name'] + str(error) + '\n'*3,
                          subject = '[%s] Auto Restart Run #%d'%(ad.solver.fea_config_dict['pc_name'], counter_auto_restart))
