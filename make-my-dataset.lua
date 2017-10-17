require ('torchx')
require ('paths')
require('image')
torch.setdefaulttensortype('torch.FloatTensor')

--cmd line arg
cmd = torch.CmdLine()
cmd:text()
cmd:text('Making train and test datasets')
cmd:text('Example:')
cmd:text('$> th make-my-dataset.lua  -path ./dataset/train/ -filename my-trainset.t7')
cmd:text('Options:')
cmd:option('-path', '/home/vik/torch/examples/dataset/train/', 'Path to train or test dataset in flat structure')
cmd:option('-filename', 'my-trainset.t7', 'Filename for output dataset. Example my-trainset.t7 - for train. my-testset.t7 - for test')

opt = cmd:parse(arg or {})
dirname = opt.path
ext = 'jpg'

--dimensions of image
imgheight = 164
imgwidth = 160


f = io.popen('ls ' .. dirname)

--creating table of labels and count number of directories
labels = {}
dir = {}
count = 0
        for name in f:lines() do
                count = count +1
                table.insert(labels, count)
                table.insert(dir, name)
        end

--creating table of files and lebeling files
imgdata = {}
ll = {} -- lebels in numbers aka 1, 2, 3
        for i=1, #dir  do

                file = paths.indexdir(dirname .. dir[i], ext, false)

                for m=1,file:size() do
                         xlua.progress(m, file:size())

                        local img = image.load(file:filename(m),3, float)

                        local scale = image.scale(img, imgwidth, imgheight)

                        table.insert(imgdata, scale)

                        table.insert(ll, i)


                end
        end


--data to tenzors
label = torch.Tensor(ll)

data = torch.Tensor(#imgdata,3, imgheight, imgwidth)

        for i=1, #imgdata do

                data[i] = imgdata[i]

        end

Datasave = {data = data, label = label}
print ("Storing training data in",opt.filename)
torch.save(opt.filename, Datasave)
