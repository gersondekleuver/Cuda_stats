import torch


class GPU_DISPLAY():

    def __init__(self):
        self.size_box_1 = 19
        self.size_box_2 = 37
        self.GPU_num = torch.cuda.device_count()
        self.CUDA_availability = torch.cuda.is_available()

    def __fit_string(self, string, fit):

        return (fit - len(str(string))) * " " + "║"

    def __color_text(self, state):
        if state:
            color = "\033[1;32;40m"
        else:

            color = "\033[1;71;31m"
        return color + str(state) + "\033[0m"

    def select_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):

        GPU_print = ""
        for GPU in range(self.GPU_num):
            newline = ""
            if(GPU == self.GPU_num + 1):
                newline = "\n"
            device = torch.cuda.get_device_name(GPU)
            capacity = torch.cuda.get_device_capability(device=GPU)

            GPU_print += f"""║ GPU {GPU}'s name:       ║   {self.__color_text(device)}{self.__fit_string(device, self.size_box_2)}\n"""
            GPU_print += f"""║ GPU {GPU}'s capacity:   ║   {self.__color_text(capacity)}{self.__fit_string(capacity, self.size_box_2)}""" + newline
        if GPU_print == "":
            GPU_print = f"""║  {self.__fit_string(GPU_print, self.size_box_1)}   {self.__fit_string(GPU_print, self.size_box_2)}"""

        return(f"""╔═════════════════════╦════════════════════════════════════════╗
║ CUDA availability:{self.__fit_string("CUDA availability", self.size_box_1)}   {self.__color_text(self.CUDA_availability)}{self.__fit_string(self.CUDA_availability, self.size_box_2)}                                      
║ CUDA_initalized:{self.__fit_string("CUDA_initalized", self.size_box_1)}   {self.__color_text(torch.cuda.is_initialized())}{self.__fit_string(torch.cuda.is_initialized(), self.size_box_2)}
║ GPU's detected:{self.__fit_string("GPU's detected", self.size_box_1)}   {self.__color_text(self.GPU_num)}{self.__fit_string(self.GPU_num, self.size_box_2)}
{GPU_print}
╚═════════════════════╩════════════════════════════════════════╝""")


gpu = GPU_DISPLAY()
print(gpu)
