
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
             self.root_dir = '/mnt/16t/laijintao/LEVIR'
            # self.root_dir = 'C:/study/data/LEVIR'
        elif data_name == 'WHU':
            self.root_dir = '/mnt/16t/laijintao/WHU'
        elif data_name == 'GZ':
            self.root_dir = '/mnt/16t/laijintao/GZ'
            # self.root_dir = 'C:/study/data/GZ'
        elif data_name == 'SYSU':
            self.root_dir = '/mnt/16t/laijintao/SYSU'
        elif data_name == 'CLCD':
            self.root_dir = '/mnt/16t/laijintao/CLCD'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

