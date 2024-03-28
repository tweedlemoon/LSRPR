import datetime
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

# 这里是Email的配置处
# 发件邮箱
EMAIL_ADDRESS = 'xxx@xxx.com'
# 发件邮箱密码
EMAIL_PASSWORD = ''
# 收件邮箱，可以扩充
RECEIVERS = ['xxx@xxx.com', ]


class EmailSender:
    def __init__(self, proName='', logAdd='', message=''):
        """
        邮件发送类
        :param proName: 调用方法os.path.basename(sys.argv[0])
        :param logAdd: 日志地址logAdd
        :param message: 先设置为空，由后面定义
        """
        # 程序名
        self.proName = proName
        # 日志地址
        self.logAdd = logAdd
        # 发送的信息
        self.message = message
        # 时间
        self.time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 上面三个预设
        self.emailSender = EMAIL_ADDRESS
        self.senderPasswd = EMAIL_PASSWORD
        self.reciever = RECEIVERS

        self.loadMessage()

    def loadMessage(self):
        """
        读入信息
        :return:self.message
        """
        if self.logAdd != '':
            with open(self.logAdd, 'r') as f:
                for line in f.readlines():
                    self.message += line

                f.close()
            return self.message
        else:
            self.message = ''
            return self.message

    def sendResultEmail(self):
        """
        发送结果邮件
        :return: Ture or False 代表发送成功与失败
        """
        ret = True
        for r in self.reciever:
            try:
                # 邮件本体
                self.getCurrentTime()
                msg = MIMEText('邮件发送时间:' + self.time + '\n' + '运行结果：' + '\n' + self.message, 'plain', 'utf-8')
                msg['From'] = formataddr(("来自您的项目" + self.proName, self.emailSender))  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
                msg['To'] = formataddr(("FK", r))  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
                msg['Subject'] = "您的程序运行完成"  # 邮件的主题，也可以说是标题

                server = smtplib.SMTP_SSL("smtp.126.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25，qq和网易为465
                server.login(self.emailSender, self.senderPasswd)  # 括号中对应的是发件人邮箱账号、邮箱密码
                server.sendmail(self.emailSender, r, msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
                server.quit()  # 关闭连接
            except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
                ret = False
        return ret

    def sendErrorEmail(self):
        """
        发送错误邮件
        :return:
        """
        ret = True
        for r in self.reciever:
            try:
                # 邮件本体
                self.getCurrentTime()
                msg = MIMEText('邮件发送时间:' + self.time + '\n' + self.message, 'plain', 'utf-8')
                msg['From'] = formataddr(("来自您的项目" + self.proName, self.emailSender))  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
                msg['To'] = formataddr(("FK", r))  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
                msg['Subject'] = "您的程序出现问题"  # 邮件的主题，也可以说是标题

                server = smtplib.SMTP_SSL("smtp.126.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25，qq和网易为465
                server.login(self.emailSender, self.senderPasswd)  # 括号中对应的是发件人邮箱账号、邮箱密码
                server.sendmail(self.emailSender, r, msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
                server.quit()  # 关闭连接
            except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
                ret = False
        return ret

    def getCurrentTime(self):
        """
        获取当前时间
        :return: none
        """
        self.time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 获取当前时间
        return


if __name__ == '__main__':
    email_sender = EmailSender(proName="runLSF", message="completed")
    email_sender.sendResultEmail()
