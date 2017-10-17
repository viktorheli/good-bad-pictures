--[[
This simple script conver you GPU cuda model to CPU model. This help to train model on GPU and after convert use the model on CPU
All parameters of you model will be saved.
--]]

require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'xlua'

--set torch tensor to float(32-bit). Gaming GPU does not support 64-bit.
torch.setdefaulttensortype('torch.FloatTensor')

--parse cmdline argument
cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert model from GPU CUDA to CPU')
cmd:text('Example:')
cmd:text('$> th convert-cuda-cpu.lua -cudamodel clear-fotograf-1000cuda-new.dat -cpumodel model-cpu.dat')
cmd:text('Options:')
cmd:option('-cudamodel', 'github.dat', 'Filename with cuda neuralnet.')
cmd:option('-cpumodel', 'github-cpu.dat', 'Filename to save converted cpu model')
cmd:option('-h', '', 'Display help')
cmd:option('-help', '', 'Display help')
opt = cmd:parse(arg or {})

--round function

function round(num, numDecimalPlaces)
  return tonumber(string.format("%." .. (numDecimalPlaces or 0) .. "f", num))
end


        if (opt.h == h or opt.help == help) then

        print (help)

        else

                if paths.filep(opt.cudamodel) == true then

                        print("----Loading CUDA model----")

                        local time = sys.clock()
                        cudanet = torch.load(opt.cudamodel)
                        local time = round((sys.clock() - time)*1000)
                        print("Loding completed in: "..time.."ms\n")

                        print("----Converting model----")
                        local time = sys.clock()
                        convert = cudnn.convert(cudanet, nn)
                        netcpu = convert:float()
                        local time = round((sys.clock() - time)*1000)
                        print("Converting completed in: "..time.."ms\n")

                        print("----Saving CPU model----")
                        local time = sys.clock()
                        torch.save(opt.cpumodel, netcpu)
                        local time = round((sys.clock() - time)*1000)
                        print("Saving CPU model completed in: "..time.."ms\n")

                        print("All done successfully. CPU model saved to file: "..opt.cpumodel)
                else
                        print("Erorr! Can not load CUDA model file!\n")
                end
        end
