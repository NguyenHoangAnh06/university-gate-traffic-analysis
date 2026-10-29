# Dữ liệu cho đề tài giao thông tại cổng trường

Thư mục này phục vụ detection, tracking đa lớp và phân tích lưu lượng từ camera cố định. Giữ nguyên dữ liệu nguồn; chưa chuyển đổi nhãn hay chia lại các tập dữ liệu.

## Năm lớp mục tiêu

Thứ tự đề xuất: `0 pedestrian`, `1 motorcycle`, `2 car`, `3 bus`, `4 truck`.

Mô tả bài toán liệt kê năm lớp ở mục 1 nhưng vẫn ghi bốn lớp ở một số mục sau. Chuẩn dữ liệu ở đây theo danh sách năm lớp ở mục 1.

| Dataset | Vai trò | Nhãn cần chú ý |
| --- | --- | --- |
| BDD100K detection | Huấn luyện/đánh giá phát hiện đối tượng | `person -> pedestrian`, `motor -> motorcycle`; giữ `car`, `bus`, `truck`. Không tự gộp `rider` vào pedestrian. |
| BDD100K MOT | Huấn luyện/đánh giá tracking đa lớp | Có ID và lớp; category có thể khác tên giữa các phiên bản. Phải kiểm tra nhãn thực tế khi chuyển đổi. |
| VisDrone2019 DET và MOT | Đối tượng nhỏ, đông đúc, che khuất | Lớp nguồn 1 pedestrian, 4 car, 6 truck, 9 bus, 10 motor. Không tự gộp `people`, `van`, bicycle hoặc tricycle vào năm lớp mục tiêu. |
| VnTraffic | Video giao thông Việt Nam, thử nghiệm tracking | Gói đã kiểm tra có video và GT tracking, nhưng không có nhãn phân loại năm lớp. Cần bổ sung nhãn lớp để đánh giá tracking theo lớp. |
| MOT20 | Benchmark tracking người đi bộ trong cảnh đông | Chỉ dùng cho pedestrian; không phải benchmark đa lớp phương tiện. |

## Cấu trúc

- `BDD100K/train/images`, `BDD100K/val/images`: ảnh detection; nhãn JSON trong thư mục `annotations` của từng split.
- `BDD100K/test`: ảnh test detection, không có GT công khai trong gói tải.
- `BDD100K/splits`: danh sách ảnh có bản ghi nhãn và ảnh thiếu bản ghi nhãn, được tạo bởi bước audit. Bản nhãn train nguồn có 69.863 bản ghi cho 70.000 ảnh, thiếu 137 bản ghi. Giữ nguyên ảnh nguồn; dùng danh sách `train_labeled_images.txt` khi chuẩn bị huấn luyện, không tự coi ảnh thiếu nhãn là ảnh âm tính.
- `BDD100K/MOT`: junction Windows tới `D:\deeplearning-datasets\BDD100K-MOT`.
- `VisDrone2019/VisDrone2019-DET-*`: ảnh và annotation detection nguyên bản. `test-challenge` không có GT công khai.
- `VisDrone2019/MOT`: junction Windows tới `D:\deeplearning-datasets\VisDrone2019-MOT`.
- `VnTraffic/VNTraffic`: video gốc, video hiển thị GT, GT TXT và JSON.
- `VnTraffic/AICC22-Custom`: bộ phụ đi cùng gói Zenodo, không phải VnTraffic.
- `MOT20`: dữ liệu có sẵn, không thay đổi.

Phần MOT lưu trên D vì ổ C không đủ dung lượng. Đường dẫn trong thư mục `data` vẫn dùng bình thường, nhưng khi sao chép sang máy khác cần sao chép cả dữ liệu đích của junction.

VnTraffic đã kiểm tra có 501 frame được gán nhãn, frame index 0–500 và 16.036 bounding box. **TXT có 85 track ID nhưng JSON có 205 track ID**; frame và tọa độ của toàn bộ 16.036 box khớp nhau giữa hai dạng, còn cách gán danh tính không nhất quán. Giữ nguyên cả hai nguồn; cần chọn và kiểm tra/chuẩn hóa một nguồn ID trước khi benchmark, không trộn TXT và JSON để tính IDF1/HOTA. Video gốc có 510 frame trong 17 giây (30 FPS); không coi các frame sau 500 là frame có GT. TXT có định dạng `frame,id,x,y,w,h,confidence,-1,-1,-1`. JSON chỉ chứa `frames` và `boxes`, không chứa class. Các đường dẫn ảnh trong JSON là đường dẫn từ môi trường tác giả; phải trích frame từ video gốc và tạo lại đường dẫn nếu dùng.

Các đường dẫn trong `BDD100K/splits/*.txt` tính từ thư mục `BDD100K/`; đây là danh sách nguồn để chuẩn bị dữ liệu, chưa phải cấu hình YOLO đã chuyển đổi.

## Nguồn và license

- BDD100K: https://github.com/bdd100k/bdd100k . License dữ liệu được lưu trong `BDD100K/ORIGINAL_LICENSE.rst`; cho phép giáo dục/nghiên cứu/phi lợi nhuận theo điều khoản. BSD của mã nguồn không thay thế license dữ liệu. Máy chủ tải chính không truy cập được trong phiên này nên lấy bản mirror; URL và checksum được lưu trong manifest.
- BDD100K detection mirror: https://huggingface.co/datasets/Myszz/BDD100k . Gói này dùng nhãn detection đời cũ, không phải nhãn Detection 2020. Giữ nguyên nhãn nguồn để tránh nhầm phiên bản thực nghiệm.
- BDD100K MOT mirror: https://huggingface.co/datasets/mrhwl123/BDD100kMOT . MD5 gói nhãn khớp checksum chính thức `6be40e0ca56a83ddeba2ed6bff50f9e6`.
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset . DET tải qua mirror Ultralytics; MOT qua https://huggingface.co/datasets/vanthanh/VisDrone2019-MOT . Chính sách tại https://aiskyeye.com/data-protection/ giới hạn học thuật và liệt kê CC BY-NC-SA 3.0 nhưng hiện ghi phiên bản 2021; cần xác minh điều khoản riêng nếu dùng ngoài phạm vi bài tập học thuật.
- VnTraffic: https://zenodo.org/records/18195750 , DOI `10.5281/zenodo.18195750`. Metadata khai báo CC BY 4.0; dữ liệu video gốc từ kênh YouTube đài truyền hình theo mô tả tác giả. Metadata nguyên bản được lưu cạnh dữ liệu.

Ghi nguồn dataset và bài báo tương ứng trong báo cáo. Không coi mirror là giấy phép sử dụng mới.

## Kiểm tra tải

Các file `*.verified.json` chỉ được tạo sau khi tải đủ, đối chiếu checksum nếu nguồn có cung cấp, và giải nén thành công với kiểm tra CRC. File `.part` là file tải dở, không được dùng làm dữ liệu hoàn chỉnh.

`download_sources.json` và `tracking_download_sources.json` ghi URL, dung lượng và checksum. File nén detection được lưu ở `D:\deeplearning-datasets\_downloads\detection`; file nén tracking nằm ở `D:\deeplearning-datasets\_downloads`. Junction `_archives` cho phép truy cập các file nén này ngay trong thư mục `data`.

Chạy `python project/dataset_status.py` từ thư mục workspace để cập nhật và xem tiến độ; kết quả máy đọc được lưu ở `download_status.json`.

Không chia ngẫu nhiên các frame cùng video sang train và test. Giữ các split chính thức khi benchmark, chia dữ liệu tự quay theo video/phiên/camera. BDD100K là camera trên xe và VisDrone là camera drone; cần đánh giá riêng trên video camera cố định tại Hà Nội để đo chênh lệch miền dữ liệu. Nhãn công khai chưa thay thế GT đếm xe theo vạch/vùng/hướng của dữ liệu thực tế.
