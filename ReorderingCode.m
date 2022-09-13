clear; clc;
lf_path = './bedroom/GT_DATA';
% lf_path = './bedroom_resi';
viewpoint_n = 9;
image_wh = 512;
flo_file = 'bedroom.flo';

LF_SAI = zeros(viewpoint_n, viewpoint_n, image_wh, image_wh, 3);
resi_idx = 1;
for y=1:viewpoint_n
    for x=1:viewpoint_n
        file = sprintf('%s/gt_%d_%d.png', lf_path, y, x);
        img  = imread(file);

%         if ((y == 1 || y == 5 || y == 9) && (x == 1 || x == 5 || x == 9))
%             img = 127*ones(image_wh, image_wh, 3);
%         else
%             file = sprintf('%s/resi_%d.png', lf_path, resi_idx);
%             img  = imread(file);
%             resi_idx = resi_idx + 1;
%         end

        LF_SAI(y, x, :, :, :) = img;
    end
end

LF_MI = SAI_to_MI(LF_SAI, viewpoint_n, image_wh); %Original MI
ReorderMap_horz = GenReorderMap(flo_file, 1, viewpoint_n, image_wh);
ReorderMap_vert = GenReorderMap(flo_file, 2, viewpoint_n, image_wh);

% buffers for reordering
LF_MI_temp1 = LF_MI;
LF_MI_temp2 = LF_MI_temp1;

% vert reordering
LF_RoMI = zeros(viewpoint_n*image_wh, viewpoint_n*image_wh, 3, 'uint8');
for u=1:size(LF_RoMI, 2)
    tar_v = (ReorderMap_vert(:, u, 1) + viewpoint_n*(ReorderMap_vert(:, u, 2)-1));
    LF_MI_temp2(:, u, :) = LF_MI_temp1(tar_v, u, :);
end
% horz reordering
for v=1:size(LF_RoMI, 1)
    tar_u = ReorderMap_horz(v, :, 1) + viewpoint_n*(ReorderMap_horz(v, :, 2)-1);
    LF_RoMI(v, :, :) = LF_MI_temp2(v, tar_u, :);
end

% Reconstruct
Recon_RoMI = zeros(viewpoint_n*image_wh, viewpoint_n*image_wh, 3, 'uint8');
% horz direction
for v=1:size(LF_RoMI, 1)
    tar_u = ReorderMap_horz(v, :, 1) + viewpoint_n*(ReorderMap_horz(v, :, 2)-1);
    LF_MI_temp1(v, tar_u, :) = LF_RoMI(v, :, :);
end
% vert direction
for u=1:size(LF_RoMI, 2)
    tar_v = ReorderMap_vert(:, u, 1) + viewpoint_n*(ReorderMap_vert(:, u, 2)-1);
    Recon_RoMI(tar_v, u, :) = LF_MI_temp1(:, u, :);
end
subplot(1, 3, 1);
imshow(LF_MI);
subplot(1, 3, 2);
imshow(LF_RoMI);
subplot(1, 3, 3);
imshow(Recon_RoMI);

mse_error = immse(LF_MI, Recon_RoMI);
fprintf("Error : %d (Error should be zero)\n", mse_error);
Recon_SAI = MI_to_SAI(LF_MI, viewpoint_n, image_wh);


%% function
function LF_MI = SAI_to_MI(LF_SAI, viewpoint_n, image_wh)
    LF_MI = zeros(size(LF_SAI, 1)*size(LF_SAI, 3), ...
        size(LF_SAI, 2)*size(LF_SAI, 4), 3, 'uint8');
    idx_v = 1;
    for v=1:image_wh
        idx_u = 1;
        for u=1:image_wh
            sub = permute(LF_SAI(:, :, v, u, :), [1, 2, 5, 3, 4]);
            LF_MI(idx_v:(idx_v+(viewpoint_n-1)), idx_u:(idx_u+(viewpoint_n-1)), :) = sub;
            idx_u = idx_u + viewpoint_n;
        end
        idx_v = idx_v + viewpoint_n;
    end
end

function ReorderMap = GenReorderMap(flo_path, direction, viewpoint_n, image_wh) % direction 1: horz, 2: vert
    flow_img = readFlowFile(flo_path); % ch1: horz_flow, ch2: vert_flow
    curr_flow = round(flow_img(:, :, direction)/4); 
    
    ReorderMap = zeros(viewpoint_n*image_wh, viewpoint_n*image_wh, 2);
    for jj=1:image_wh
        tracer = ones(1, viewpoint_n);
        selc_line = zeros(0);
        if(direction == 1)
            flow_line = curr_flow(jj, :);
        elseif(direction == 2)
            flow_line = curr_flow(:, jj)';
        end
        
        %disparity window (for smoothing)
        window_size = 9;
        flow_line_new = flow_line;
        for ii=1:size(flow_line, 2)
            s = ii - (window_size - 1)/2;
            if(s < 1); s = 1;   end
            e = ii + (window_size - 1)/2;
            if(e > image_wh); e = image_wh;   end

            flow_line_new(1, ii) = mean(flow_line(1, s:e));
        end
        flow_line = round(flow_line_new);
        
        for ii = 1:image_wh
            pel_pos = (flow_line(1, ii) * ((0:(viewpoint_n-1)) - ((viewpoint_n-1)/2)) + ii);

            % fill pels
            while(sum((pel_pos > tracer) & (pel_pos <= image_wh)) >= 1)
                for v_pos = 1:viewpoint_n
                    if((pel_pos(1, v_pos) > tracer(1, v_pos)) && tracer(1, v_pos) < image_wh)
                        selc_line = cat(2, selc_line, cat(1, v_pos, tracer(1, v_pos)));
                        tracer(1, v_pos) = tracer(1, v_pos) + 1;
                    end
                end
            end
            
            for v_pos = 1:viewpoint_n
                if ((pel_pos(1, v_pos) == tracer(1, v_pos)) && tracer(1, v_pos) < image_wh)
                    selc_line = cat(2, selc_line, cat(1, v_pos, pel_pos(1, v_pos)));
                    tracer(1, v_pos) = tracer(1, v_pos) + 1;
%                 else
%                     assert(1);
                end
            end
        end
        
        % fill left pels
        pel_pos = (image_wh+1)*ones(1, viewpoint_n);
        while(sum(tracer == (image_wh+1)) ~= viewpoint_n)
            for v_pos = 1:viewpoint_n
                if((pel_pos(1, v_pos) > tracer(1, v_pos)) && tracer(1, v_pos) < (image_wh+1))
                    selc_line = cat(2, selc_line, cat(1, v_pos, tracer(1, v_pos)));
                    tracer(1, v_pos) = tracer(1, v_pos) + 1;
                end
            end
        end
        ReorderMap((viewpoint_n*(jj-1)+1):(viewpoint_n*(jj-1)+viewpoint_n), :, 1) = repmat(selc_line(1, :), [viewpoint_n, 1]);
        ReorderMap((viewpoint_n*(jj-1)+1):(viewpoint_n*(jj-1)+viewpoint_n), :, 2) = repmat(selc_line(2, :), [viewpoint_n, 1]);
        
        debug_table = zeros(viewpoint_n, image_wh, 'uint8');
        for t_idx=1:size(selc_line, 2)
            debug_table(selc_line(1, t_idx), selc_line(2, t_idx)) = debug_table(selc_line(1, t_idx), selc_line(2, t_idx)) + 1;
        end
        assert(sum(debug_table(:) ~= 1) == 0);
    end
    
    if(direction == 2)
        ReorderMap(:, :, 1) = ReorderMap(:, :, 1)';
        ReorderMap(:, :, 2) = ReorderMap(:, :, 2)';
    end
end

function LF_SAI = MI_to_SAI(LF_MI, viewpoint_n, image_wh)
    LF_SAI = zeros(viewpoint_n, viewpoint_n, image_wh, image_wh, 3, 'uint8');
    idx_v = 1;
    for v=1:image_wh
        idx_u = 1;
        for u=1:image_wh
            sub = LF_MI(idx_v:(idx_v+(viewpoint_n-1)), idx_u:(idx_u+(viewpoint_n-1)), :);
            LF_SAI(:, :, v, u, :) = sub;
            idx_u = idx_u + viewpoint_n;
        end
        idx_v = idx_v + viewpoint_n;
    end
        
end

