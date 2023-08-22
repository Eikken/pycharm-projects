def login(self):
    self.driver.switch_to.frame('login_frame')
    self.driver.find_element_by_id('switcher_plogin').click()
    self.driver.find_element_by_id('u').clear()
    self.driver.find_element_by_id('u').send_keys(self.__username)
    self.driver.find_element_by_id('p').clear()
    self.driver.find_element_by_id('p').send_keys(self.__password)
    self.driver.find_element_by_id('login_button').click()
    self.driver.get('http://user.qzone.qq.com/{}'.format(self.__username))
    cookie = ''
    for item in self.driver.get_cookies():
        cookie += item["name"] + '=' + item['value'] + ';'
    self.cookies = cookie
    self.headers['Cookie'] = self.cookies
    self.driver.quit()
