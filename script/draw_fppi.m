clear results
clear gts
fid = fopen("./name.txt");
data = textscan(fid, '%s', 'delimiter', '\n');
data = data{1,1};
[length,n] = size(data);
fid_anno = fopen("./anno.txt");
data_anno = textscan(fid_anno, '%s', 'delimiter', '\n');
data_anno = data_anno{1,1};
fid_score = fopen("./score.txt");
data_score = textscan(fid_score, '%s', 'delimiter', '\n');
data_score = data_score{1,1};
fid_gt = fopen("./gt.txt");
data_gt = textscan(fid_gt, '%s', 'delimiter', '\n');
data_gt = data_gt{1,1};


results(length) = struct('Boxes', [], 'Score', []);
gts(length) = struct('Boxes', []);
for i =1:length
    A = data_anno(i);
    A = str2num(cell2mat(A));
    results(i).Boxes = A;
    B = data_score(i);
    B = str2num(cell2mat(B));
    results(i).Score = B;
    C = data_gt(i);
    C = str2num(cell2mat(C));
    gts(i).Boxes = C;
end
results = struct2table(results);
gts = struct2table(gts);
[am, fppi, missRate] = evaluateDetectionMissRate(results, gts(:, 1));

figure
loglog(fppi, missRate);
grid on
title(sprintf('log Average Miss Rate = %.5f',am))
