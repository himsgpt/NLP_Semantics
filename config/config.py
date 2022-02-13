# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 12:02:15 2019

@author: higupta
"""

import configparser
config = configparser.ConfigParser()
config["Propsal_action_1st_step"] = {'mcp_xpath': '//*[@id="app"]/div/div[1]/div[2]/div[2]/div/table/tbody/tr[4]/td[1]/div/ul/li/span[1]',
                                     'mcp_click_xpath': '//*[@id="app"]/div/div[1]/div[2]/div[2]/div/table/tbody/tr[4]/td[3]/div/div[2]/h3/span'}

with open('EP_Automation\config\Html_paths.ini', 'w') as configfile:
      config.write(configfile)

"""
import configparser
config = configparser.ConfigParser()
config.read('EP_Automation\config\Html_paths.ini')
config.sections()
config['Propsal_action_1st_step']['mcp_xpath']