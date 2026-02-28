/**
  ******************************************************************************
  * @author		Anton Houzich
  * @version	V2.0.0
  * @date		29-April-2023
  * @mail		houzich_anton@mail.ru
  * discussion  https://t.me/brute_force_gpu
  ******************************************************************************
  */
#include <stdafx.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <Dispatcher.h>
#include <thread>

int main()
{
    return Generate_Mnemonic();
}