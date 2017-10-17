--[[
This script loads the neural network into memory. Listens to the tcp port
and waits for the image file to load. Next, the image passes through the neural network,
which gives a good or bad response for image.
--]]

require 'cunn'
require 'cutorch'
require 'image'
--enable waffle module
app = require 'waffle'

--images dimensions. Must match what is specified in the neural network
imgheight = 164
imgwidth = 160

--categories
classes = {'bad', 'good'}


--round function

function round(num, numDecimalPlaces)
  return tonumber(string.format("%." .. (numDecimalPlaces or 0) .. "f", num))
end


--loading neuralnetwork in memory

                net = torch.load("github-clear.dat")


--define simple web form to browse and upload file

        app.get('/', function(req, res)
           res.send(html { body { form {
                method = 'POST',
                enctype = 'multipart/form-data',
                p { input {
                type = 'file',
                name = 'file'
                }},
                p { input {
                type = 'submit',
                'Upload'
                }}
                }}})
        end)

--Processing request from client


        app.post('/', function(req, res)

--upload image
 time = sys.clock()
                img = nill
                img = req.form.file:toImage()

-- if file not load or incorrect file format
                if img == nill or img:size(1) ~=3 then
                        print("kosyak!!!")
                        res.send("Incorrect file type.")
                else
--load image, scale it and forwarding througth neural net
                load = img
                imgscaled = image.scale(load, imgwidth, imgheight)
                imgdata = torch.Tensor(imgscaled,3, imgwidth, imgheight):cuda()
                predict = net:forward(imgdata):cuda()

                predicted = predict:exp()*100

--define temporary lua table for predicted data

mytable = ""

--Do the matching category and predicted probability and concatenate to table. For example bad: 40 good: 60.

                for i=1,predicted:size(1) do

                mytable = mytable..classes[i]..": "..round(predicted[i],3).."\n "

                end

--send data to client
        res.send(mytable)

print (mytable)
                end
                 time = sys.clock() - time
                print("Time to processing:"..round(time*1000).."ms")
end)

--permanent listening tcp port for connections

app.listen({host='0.0.0.0', port=8787})
