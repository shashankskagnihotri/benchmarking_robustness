# CFNet Metrics

scalar_outputs = {"loss": loss}
image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
if compute_metrics:
    with torch.no_grad():
        image_disp_error = DispErrorImageFunc()
        image_outputs["errormap"] = [image_disp_error.forward(disp_est, disp_gt) for disp_est in disp_ests]
        scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
        scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
        scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

# GWCNet Metrics
scalar_outputs = {"loss": loss}
image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
if compute_metrics:
    with torch.no_grad():
        image_disp_error = DispErrorImageFunc()
        image_outputs["errormap"] = [image_disp_error.apply(disp_est, disp_gt) for disp_est in disp_ests]
        scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
        scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
        scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]


# STTR / STTR light

summary.writer.add_scalar(mode + '/rr', stats['rr'], epoch)
summary.writer.add_scalar(mode + '/l1', stats['l1'], epoch)
summary.writer.add_scalar(mode + '/l1_raw', stats['l1_raw'], epoch)
summary.writer.add_scalar(mode + '/occ_be', stats['occ_be'], epoch)
summary.writer.add_scalar(mode + '/epe', stats['epe'], epoch)
summary.writer.add_scalar(mode + '/iou', stats['iou'], epoch)
summary.writer.add_scalar(mode + '/3px_error', stats['px_error_rate'], epoch)


