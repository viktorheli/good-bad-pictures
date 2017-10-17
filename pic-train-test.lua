require 'paths'
require 'image'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'cutorch'


torch.setdefaulttensortype('torch.FloatTensor')

--cmd line arg
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification')
cmd:text('Example:')
cmd:text('$> th 111.lua --train 10')
cmd:text('Options:')
cmd:option('-batchsize', 5, 'size of batch for training.')
cmd:option('-img', 'none', 'path to image for test')
cmd:option('-save', '.', 'path to save. Default local directory')
cmd:option('-storenet', 'my-network.dat', 'File name to stote training data')
cmd:option('-clearnet', 'my-clear-network.dat', 'File name to stote clear net data. It reduce file size, but clear training data.')
cmd:option('-train', 0, 'train iterations')
cmd:option('-trainset', 'for-github.t7', 'Dataset to train network')
cmd:option('-testset', 'for-github-test.t7', 'Dataset to train network')
cmd:option('-imgdisp', 'no', 'Display or not display image')

opt = cmd:parse(arg or {})

--round function

function round(num, numDecimalPlaces)
  return tonumber(string.format("%." .. (numDecimalPlaces or 0) .. "f", num))
end

--dimensions of image
imgheight = 164

imgwidth = 160

--image classes
classes = {'bad', 'good'}


--loading trainset
if (opt.train > 0) then
                trainset = torch.load(opt.trainset)

                        setmetatable(trainset,
                        {__index = function(t, i)
                            return {t.data[i], t.label[i]}
                        end}
                        );

                trainset.data = trainset.data:float():cuda() -- convert the data from a ByteTensor to a DoubleTensor.

--loading testset

                testset = torch.load(opt.testset)

                setmetatable(testset,
                        {__index = function(t, i)
                            return {t.data[i], t.label[i]}
                        end}
                        );


                testset.data = testset.data:float():cuda() -- convert the data from a ByteTensor to a DoubleTensor.


        function trainset:size()
                return self.data:size(1)
        end


        function testset:size()
                return self.data:size(1)
        end
end






        if (paths.filep(opt.img) == true and paths.filep(opt.clearnet) ) then

                net = torch.load(opt.clearnet):cuda()
                img = opt.img
                load = image.load(img, 3, float)

                imgscaled = image.scale(load, imgwidth, imgheight)

                --to torch tensor
                imgdata = torch.Tensor(imgscaled,3, imgwidth, imgheight):cuda()


--forward image through net
                predict = net:forward(imgdata):cuda()

                predicted = predict:exp()*100

                for i=1,predicted:size(1) do

                        print(classes[i],round(predicted[i]))
                end
                if (opt.imgdisp == "yes" ) then
                        image.display(imgscaled)
                        collectgarbage()
                end
                else
--                      print("Warning!!! No image or pretrained model found!")

        end

-- if pretrained net is exist, load them, if not exits create network
        if (opt.train > 0) then

                if (paths.filep(opt.storenet)) then
                        net = torch.load(opt.storenet)
                else

                        print('No pretrained model found. Create new net, and train it. ')

--Creating convolutonal network

                net = nn.Sequential()

                net:add(nn.SpatialConvolution(3, 128, 4, 4))
                net:add(nn.Tanh())
                net:add(nn.SpatialMaxPooling(2,2,2,2))
                net:add(nn.SpatialConvolution(128, 256, 5, 5))
                net:add(nn.Tanh())
                net:add(nn.SpatialMaxPooling(2,2,2,2))
                net:add(nn.SpatialConvolution(256, 512, 5, 5))
                net:add(nn.Tanh())
                net:add(nn.SpatialMaxPooling(2,2,2,2))
                net:add(nn.View(512*17*16))
                net:add(nn.Linear(512*17*16, 120))
                net:add(nn.Tanh())
                net:add(nn.Linear(120, 84))
                net:add(nn.Tanh())
                net:add(nn.Linear(84, 24))
                net:add(nn.Tanh())
                net:add(nn.Linear(24, 2))
                net:add(nn.LogSoftMax())
                net:cuda()
                print(net)
        end


--Using optim for training

        trainSize = trainset.data:size(1)               --size of training data
        testSize = testset.data:size(1)                 --size of test data

        loss = nn.ClassNLLCriterion():cuda()



        local optimState = {
                learningRate = 1e-2,
                momentum = 0.5,--0.1,
                weightDecay = 0.0005--1e-5
                }


        batchSize = opt.batchsize
        print (os.date())
        print ('Quantity of training epochs: '..opt.train)
        print ('Quantity of training images: '..trainSize)
        print ('Quantity of testing images: '..testSize)
        print ('Batch size: '..opt.batchsize..'\n')


        local x = torch.CudaTensor(batchSize, trainset.data:size(2), trainset.data:size(3), trainset.data:size(4))
        local yt = torch.CudaTensor(batchSize)


        local w,dE_dw = net:getParameters()

        local confusion = optim.ConfusionMatrix(classes)

        local function train()

                 shuffle = torch.randperm(trainSize)

        for t = 1, trainSize, batchSize do
                xlua.progress(t, trainSize)

        -- batch fits?
        if (t + batchSize - 1) > trainSize then
            break
        end

    -- create mini batch
        local idx = 1
        for i = t, t + batchSize - 1  do

                x[idx] = trainset.data[shuffle[i]]
                yt[idx] = trainset.label[shuffle[i]]
            if yt[idx] == 0 then
                yt[idx] = 1
            end
            idx = idx + 1
        end


   -- create f(X) and df/dX
        local eval_E = function(w)
                        collectgarbage()
                --get new parameters

                if w ~= w then
                    w:copy(w)
                end

            -- reset gradients
            dE_dw:zero()

            -- evaluate function for complete mini batch
            local y = net:forward(x)
                y:cuda()
             E = loss:forward(y,yt)

            local dE_dy = loss:backward(y,yt)
            net:backward(x,dE_dy)

            return E, dE_dw

        end

 -- optimize on current mini-batch
                err = optim.sgd(eval_E, w, optimState)

    end

end


                local time = sys.clock()

                for i = 1, opt.train do
--periodically save model. Every 1000 epochs

                        if i % 1000 == 0 then
                                print ('\nSavind temporal model to:', opt.storenet ..'-temporal')
                                torch.save(opt.storenet.."-temporal", net)
                                print ('\nSavind temporal model with clearstate to:', opt.clearnet ..'-temporal')
                                torch.save(opt.clearnet.."-temporal", net:clearState())
                        end
                        print ('\nDo epoch: '.. i)
                        train()
                        print('Error for epoch '..i..': '.. err[1]..' ')
                        print('Loss for epoch '..i..': '.. E..' ')
                end
        time = sys.clock() - time
        print(time .. ' seconds needed to '..opt.train..' epochs of training\n')


-- test over test data

        time = sys.clock()
                print('==> testing on test set:')
        for t = 1,testSize do

    -- disp progress
                xlua.progress(t, testSize)

    -- get new sample
                local input = testset.data[t]
                local target = testset.label[t]

    -- test sample
                local pred = net:forward(input):view(2)

                confusion:add(pred, target)
        end

-- timing
time = sys.clock() - time
time = time / testSize
print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

-- print confusion matrix
print(confusion)
print (os.date())

--Saving full net for further training


                print ("Storing training data in",opt.storenet)

                torch.save(opt.storenet, net)

                print ("Storing clear net in",opt.clearnet)

                torch.save(opt.clearnet, net:clearState())
end
