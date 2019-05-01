NeuralNetwork = {}

function NeuralNetwork.visualise(nn)
    local function WBcolor(value) --weight/bias-color
        if value > 0 then
            return {0, value, 0, 1}
        elseif value < 0 then
            return {-value, 0, 0, 1}
        end
    end

    --calculate distances:
    local diameterx = w/(#nn+1)
    local diametery = h
    for k, v in ipairs(nn) do
        local dis = h/(#v+1)
        diametery = math.min(diametery, dis)
    end
    local radius = math.min(diameterx, diametery)/4 --radius of circles in neural network
    draw.setfont('menlo', radius/2)

    draw.beginframe()
    draw.clear(colors.black)
    for lay = #nn,1,-1 do
        local x1 = w/(#nn+1)*lay
        local x2 = w/(#nn+1)*(lay-1)
        for i, neuron in ipairs(nn[lay]) do
            local y1 = h/(#nn[lay]+1)*i
            if lay ~= 1 then
                --draw line:
                for j = 1,#neuron.weights do
                    local y2 = h/(#nn[lay-1]+1)*j
                    draw.line(x1, y1, x2, y2, WBcolor(neuron.weights[j]))
                end
                --draw color of bias:
                draw.fillcircle(x1, y1, radius, WBcolor(neuron.bias))
            end
            --draw neuron:
            draw.fillcircle(x1, y1, radius, colors.black)
            draw.circle(x1, y1, radius, colors.white)
            --draw text:
            draw.stringinrect(lay..'/'..i, x1-radius, y1-radius*0.6, x1+radius, y1+radius, colors.white)
            draw.stringinrect('\n'..(neuron.bias or ''), x1-radius, y1-radius*0.6, x1+radius, y1+radius, colors.white)
        end
    end
    draw.endframe()
end

function NeuralNetwork.CreateNetwork(num_neurons, learning_factor)
    local nn = {learning_factor=learning_factor}

    for lay = 1,#num_neurons do
        nn[lay] = {}
        for i = 1,num_neurons[lay] do
            nn[lay][i] = {sum=false, a_sum=false, weights={}, bias=false, dBias=false, dWeights={}, dCost=false}
            if lay ~= 1 then
                nn[lay][i].bias = math.random()*2-1
                for k = 1,num_neurons[lay-1] do
                    nn[lay][i].weights[k] = math.random()*2-1
                end
            end
        end
    end

    return nn
end

function sigmoid(z)
    return 1/(1+math.exp(-z))
end

function dSigmoid(z)
    return sigmoid(z)*(1-sigmoid(z))
end

function NeuralNetwork.FeedForward(nn)
    for lay = 2,#nn do
        for i, neuron in ipairs(nn[lay]) do
            local sum = neuron.bias
            for j = 1,#nn[lay-1] do
                sum = sum + nn[lay-1][j].a_sum*neuron.weights[j]
            end
            neuron.sum = sum
            neuron.a_sum = sigmoid(sum)
        end
    end

    return nn
end

function NeuralNetwork.GetCost(nn, ExpOutputs)
    local cost = 0
    for i, neuron in ipairs(nn[#nn]) do
        cost = cost + (neuron.a_sum-ExpOutputs[i])^2
    end
    cost = cost/(2*#nn)

    return cost
end

function NeuralNetwork.BackPropagation(nn, ExpOutputs)
    for lay = #nn,2,-1 do
        for i, neuron in ipairs(nn[lay]) do
            if lay == #nn then
                neuron.dCost = (neuron.a_sum-ExpOutputs[i])/#nn[#nn]
            else
                neuron.dCost = 0
                for nextI, nextNeuron in ipairs(nn[lay+1]) do
                    neuron.dCost = neuron.dCost +
                    nextNeuron.dCost *
                    dSigmoid(nextNeuron.sum) *
                    nextNeuron.weights[i]
                end
            end

            neuron.dBias = dSigmoid(neuron.sum)*neuron.dCost
            for j = 1,#neuron.weights do
                neuron.dWeights[j] = neuron.dBias*nn[lay-1][j].a_sum
            end
        end
    end

    return nn
end

function NeuralNetwork.train(nn)
    for lay = 2,#nn do
        for i, neuron in ipairs(nn[lay]) do
            --new bias:
            neuron.bias = neuron.bias -neuron.dBias*nn.learning_factor
            --new weights:
            for j = 1,#neuron.weights do
                neuron.weights[j] = neuron.weights[j] -neuron.dWeights[j]*nn.learning_factor
            end
        end
    end

    return nn
end

function BytesToNumber(text, a, b) --This function converts the ASCII-characters into numbers from 0 to 255 and then converts those numbers one big number.
    local bytes = {}
    for i = a, b do
        table.insert(bytes, text:sub(i, i):byte())
    end
    local rn = 0 --return number
    for i, byte in ipairs(bytes) do
        rn = rn + byte*2^(8*(b-a-i+1))
    end
    return rn
end

local status = 'train' --status of training, 'train' OR status of testing, 't10k'
local file = io.open('HandwrittenNumbers/'..status..'-images.png', 'r')
local images = file:read('*a')
file:close()
local file = io.open('HandwrittenNumbers/'..status..'-labels.png', 'r')
local labels = file:read('*a')
file:close()

local samples = BytesToNumber(images, 5, 8) --de 5e tot en met de 8e byte van de file bevat het aantal samples in de database
local rows = BytesToNumber(images, 9, 12) --de 9e tot en met de 12e byte van de file bevat het aantal rijen in één van de images
local columns = BytesToNumber(images, 13, 16) --de 13e tot en met de 16e byte van de file bevat het aantal kolommen in één van de images

local FileRequested = 'w&b/'..sys.input('What do you want to name the file in which the neural network will be saved?\nw&b/')
if io.open(FileRequested) then
	error('This name is already occupied by another file!')
end

local nn = NeuralNetwork.CreateNetwork({rows*columns, 100, 10}, 1)
local file = io.open(FileRequested, 'w')
file:write('--sample #0\nlocal nn = {}\nnn.learning_factor = '..nn.learning_factor..'\n\nreturn nn')
file:close()

local UltimateTest
if status == 't10k' then
    UltimateTest = {right=0, wrong=0}
end

draw.setscreen(1)
w, h = draw.getport()

-------------------------MAINLOOP-------------------------

for sample = 1, samples do
	--read the right label:
	local label = labels:sub(8+sample, 8+sample):byte()

	--insert inputs:
	for input = 1, rows*columns do
		local pos = 16+input+(sample-1)*rows*columns
		nn[1][input].a_sum = images:sub(pos, pos):byte()/255
	end

    --make ExpOutputs an array:
    local ExpOutputs = {}
    for i = 1,10 do
        if i-1 == label then
            ExpOutputs[i] = 1
        else
            ExpOutputs[i] = 0
        end
    end

    --feed forward:
    nn = NeuralNetwork.FeedForward(nn)

    --calculate cost and draw performance:
    local cost = NeuralNetwork.GetCost(nn, ExpOutputs)
    local prediction = {index=false, value=0}
    for k, v in ipairs(nn[#nn]) do
        if v.a_sum > prediction.value then
            prediction = {index=k-1, value=v.a_sum}
        end
    end
    if prediction.index == label then
        draw.point(w/samples*sample, h-cost*h, colors.green)
        if status == 't10k' then
            UltimateTest.right = UltimateTest.right + 1
        end
    else
        draw.point(w/samples*sample, h-cost*h, colors.red)
        if status == 't10k' then
            UltimateTest.wrong = UltimateTest.wrong + 1
        end
    end
    draw.settitle(sample)

    if status == 'train' then
        --differentiate the neural network and change the weights and biases accordingly:
        nn = NeuralNetwork.BackPropagation(nn, ExpOutputs)
        nn = NeuralNetwork.train(nn)

        --save nn every 1.000 samples:
        if sample%1000 == 0 then
            draw.settitle('Saving data...')
            local file = io.open(FileRequested, 'w')
            file:write('--sample #'..(beginsample%samples+sample-1)..'\nlocal nn = {}\nnn.learning_factor = '..nn.learning_factor..'\n')
            for lay = 1, #nn do
                file:write('nn['..lay..'] = {}\n')
                for i, neuron in ipairs(nn[lay]) do
                    file:write('nn['..lay..']['..i..'] = {bias='..tostring(neuron.bias)..', dBias={}, weights={'..table.concat(neuron.weights, ',')..'}, dWeights={}}\n')
                end
            end
            file:write('\nreturn nn\n')
            file:close()
        end
    end
end

draw.settitle(status..'-images.png finished!'..(status=='t10k' and ' Touch the screen to see results' or ''))

if status == 't10k' then
    draw.waittouch()
    --draw results on the text screen:
    draw.setscreen(0)
    print('amount of digits right: '..UltimateTest.right..' ('..UltimateTest.right*100/samples..'%)')
    print('amount of digits wrong: '..UltimateTest.wrong..' ('..UltimateTest.wrong*100/samples..'%)')
end

while true do end
