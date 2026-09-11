# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

> **Ghi chú về môi trường chạy:** mình không có key OpenAI trả phí nên cấu hình
> `.env` trỏ sang endpoint tương thích OpenAI của Google Gemini
> (`https://generativelanguage.googleapis.com/v1beta/openai/`), với
> `LAB_MODEL=gemini-3.5-flash` đóng vai model lớn và
> `LAB_MINI_MODEL=gemini-3.1-flash-lite` đóng vai model nhỏ. Code trong
> `template.py` không sửa dòng nào vì OpenAI SDK tự đọc `OPENAI_BASE_URL`.
> Mọi số liệu dưới đây là kết quả mình chạy thật, không phải suy đoán.

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Trước khi chạy mình đinh ninh sẽ thấy một đường cong đẹp kiểu "temperature càng cao càng đa dạng", nhưng số liệu thật thì bướng hơn nhiều: cả bốn lần gọi (0.0 → 7.32s/105 từ, 0.5 → 6.31s/254 từ, 1.0 → 6.23s/159 từ, 1.5 → 6.32s/208 từ) đều kể đúng một chuyện là hang Sơn Đoòng, với gần như cùng bộ ba ý về kích thước, hệ sinh thái riêng và chuyện ông Hồ Khanh tìm ra hang, nghĩa là temperature đổi cách diễn đạt chứ không đẩy model sang một sự thật khác — khi một đáp án đã có xác suất trội hẳn thì nới phân phối lấy mẫu vẫn chưa đủ lật ngược lựa chọn đó. Độ dài cũng không tăng đều theo temperature (105 → 254 → 159 → 208 từ, lên xuống lộn xộn) và độ trễ thì phẳng lì quanh 6–7 giây ở mọi mức, nên mình bỏ luôn ý định dùng temperature để điều khiển độ dài hay tốc độ. Điều làm mình bất ngờ nhất là khi gọi lại lần thứ hai ở đúng temperature 0.0: model trả về một bản văn khác hẳn, lần đầu viết "chứa vừa một khu phố của thành phố New York" còn lần sau đổi thành "một khu phố ở Manhattan... hoặc có thể xếp vừa 68 chiếc máy bay Boeing 747", và tệ hơn là năm phát hiện hang mâu thuẫn giữa các lần chạy — bản 0.5 ghi 1991 còn bản 0.0 lần hai ghi 1990. Vậy temperature 0 chỉ làm phân phối nhọn hơn chứ tuyệt nhiên không bảo đảm tất định, phần không xác định còn lại đến từ hạ tầng phục vụ như thứ tự phép tính dấu phẩy động và batching trên GPU; bài học mình rút ra là muốn tái lập chính xác thì phải cache kết quả chứ đặt `temperature=0` là chưa đủ, và việc một chi tiết đơn giản như năm tháng vẫn sai lệch giữa hai lần gọi nhắc mình đừng tin phản hồi model ở chỗ cần độ chính xác dữ kiện. *(Số "từ" đếm bằng `len(text.split())` chứ không dùng `count_tokens`, vì tiktoken không có bảng mã cho Gemini nên hàm rơi về ước lượng `len(text)//4`; ngoài ra vài phản hồi bị cắt ở `max_tokens=1200` do Gemini 3.x tiêu token cho suy luận nội bộ, nên so sánh độ dài chỉ mang tính tham khảo.)*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình chọn 0.0–0.2, nhưng thí nghiệm lại dạy mình một điều ngược với lý do mình định viết ban đầu. Mình dựng một tình huống hỗ trợ thật: system prompt ghim sự kiện "chính sách đổi trả của shop là 7 ngày", rồi hỏi cùng một câu ba lần ở temperature 0.0 và ba lần ở 1.5. Kết quả là cả sáu lần đều trả lời đúng 7 ngày, câu chữ gần như không phân biệt được ("Dạ, chính sách đổi trả của bên em là trong vòng **7 ngày** kể từ khi nhận hàng ạ."), kể cả ở 1.5. Tức là khi system prompt đã ghim dữ kiện và câu hỏi đủ hẹp thì temperature gần như mất tác dụng — thứ giữ cho câu trả lời nhất quán là *system prompt*, không phải temperature. Mình vẫn để temperature thấp, nhưng giờ hiểu nó là lớp phòng thủ thứ hai chứ không phải thứ nhất: nó chỉ thực sự có ý nghĩa ở những câu mở, nơi model phải tự chọn nội dung, còn với câu đóng đã có đáp án ghim sẵn thì đặt bao nhiêu cũng vậy. Đổi lại, rủi ro thật của chatbot hỗ trợ nằm ở chỗ khác: nếu không ghim chính sách vào system prompt, model sẽ bịa ra một con số nghe rất hợp lý ở bất kỳ temperature nào.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Mình tính thẳng bằng `PRICING_PER_1K_TOKENS` trong code: 30.000 lượt gọi mỗi ngày × 350 token đầu ra = 10,5 triệu token/ngày, ra $105,00/ngày cho GPT-4o và $6,30/ngày cho mini, tức **16,67 lần** — quy ra $3.150 so với $189 mỗi tháng, chênh $2.961/tháng hay $36.025 mỗi năm. Một chi tiết mình chỉ nhận ra khi in cả hai tỉ lệ: giá input cũng chênh đúng 16,67 lần (0,0025 so với 0,00015), nên tỉ lệ này không đổi dù tỉ trọng input/output có thế nào — con số 16,67 là đặc tính của bảng giá chứ không phải của workload. GPT-4o xứng đáng khi sai một lần là tốn tiền thật hoặc mất uy tín, ví dụ trích xuất điều khoản từ hợp đồng để đưa vào hệ thống thanh toán, vì $36.000/năm rẻ hơn nhiều so với một lần đọc sai điều khoản phạt. Ngược lại mini là lựa chọn hiển nhiên cho những việc lặp đi lặp lại và dễ kiểm chứng như phân loại ticket vào 5 nhóm có sẵn, gợi ý tiêu đề, hay tóm tắt hội thoại — mình đã dùng đúng ý này ở Câu 4.2 khi đề xuất gọi `call_openai_mini` cho bước tóm tắt history.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Mình đoán bản cho trẻ 8 tuổi sẽ ngắn hơn hẳn, và đoán sai: bản giáo viên 450 từ trong 9,58s còn bản chuyên gia 503 từ trong 10,26s, chênh vỏn vẹn 12%. Nhưng số ký tự lại chênh tới 30% (2.044 so với 2.657), và chính khoảng lệch giữa hai tỉ lệ đó mới là thứ đáng nói: bản chuyên gia không trình bày *nhiều ý hơn*, nó chỉ dùng *từ dài hơn*, vì một cụm như "Elliptic Curve Digital Signature Algorithm" ngốn rất nhiều ký tự cho đúng một khái niệm. Về từ vựng thì hai bên ở hai thế giới khác nhau — bản giáo viên không có lấy một thuật ngữ tiếng Anh nào ngoài chính chữ Block và Chain, mà còn giải nghĩa ngay bằng hình ảnh "một trang giấy" và "sợi dây xích nối trang mới vào trang cũ", trong khi bản chuyên gia dày đặc DLT, P2P, Trustless, Immutable, Merkle Root, Nonce, Previous Hash, ECDSA, PoW, KYC. Ví dụ cũng đổi theo vai: giáo viên dựng hẳn một câu chuyện cả lớp đổi thẻ hình siêu nhân rồi bạn Nam tẩy xóa sổ để đòi thêm đồ chơi và bị cả lớp đối chiếu phát hiện, còn model tự xưng "cô", gọi người đọc là "con" và kết bằng một câu mời hỏi thêm — toàn bộ những chi tiết đó không hề có trong system prompt mà model tự suy ra từ hai chữ "giáo viên tiểu học", và đó là điều mình thấy ấn tượng nhất. Điểm mấu chốt là nội dung nền không đổi: cả hai đều nói đúng một ý rằng sổ cái được nhân bản cho nhiều người giữ và các khối nối nhau nên sửa một chỗ là lộ ngay, chỉ khác ở lớp mã hóa — bên thì "sợi xích bị đứt", bên thì "Previous Hash tạo liên kết chuỗi tuyến tính chống giả mạo". Nói cách khác system prompt không thêm kiến thức cho model, nó thu hẹp không gian phong cách và đặt giả định về trình độ người đọc, nên đây là cách rẻ nhất để định hình sản phẩm: đổi vài chục token là đổi toàn bộ trải nghiệm mà không cần fine-tune. *(Bản chuyên gia bị cắt giữa chừng ở chữ "tiêu t" do chạm `max_tokens=2000`, nhắc mình rằng persona càng đòi chi tiết kỹ thuật thì càng ngốn token, nên system prompt và `max_tokens` phải thiết kế cùng nhau.)*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn tiếng Việt 123 từ của mình cho `count_tokens` 151 token, trong khi công thức `số từ / 0.75` ước 164, tức **lệch −7,9%** — ước lượng thô đoán dư. Để có cái so sánh mình dịch đúng đoạn đó sang tiếng Anh (89 từ, 99 token, ước 118,7, lệch −16,6%) và chỗ này mới lộ ra điều thú vị: nếu tính theo *từ* thì tiếng Việt chỉ tốn nhiều hơn một chút (1,23 so với 1,11 token/từ), nhưng tính theo *ký tự* thì khoảng cách vọt lên hẳn (0,275 so với 0,168 token/ký tự, tức gấp 1,64 lần). Lý do là bộ mã hóa được huấn luyện chủ yếu trên tiếng Anh nên nó gộp trọn những từ Anh thông dụng thành một token, còn tiếng Việt vừa bị chẻ nhỏ theo âm tiết vừa phải trả thêm cho dấu thanh — các ký tự như "ế", "ữ", "ợ" không nằm trong bảng mã cơ sở nên bị tách thành nhiều byte. Mình cũng thử ép model lạ để xem nhánh dự phòng chạy ra sao: `count_tokens(vi, model="gemini-3.5-flash")` rơi về `len(text)//4` và trả 137 so với 151 thật, lệch khoảng 9% — chấp nhận được để ước tính chi phí, nhưng không đủ chính xác để canh sát giới hạn ngữ cảnh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Mình định viết câu kinh điển "streaming làm chữ đầu hiện gần như tức thì" thì đo thật và phải viết lại: với cùng một prompt, bản không streaming chờ 4,13s rồi hiện trọn 384 ký tự, còn bản streaming hiện chữ đầu sau 3,80s và chạy hết trong 4,27s — chỉ sớm hơn vỏn vẹn **0,33 giây**, mà tổng thời gian lại *chậm hơn* 0,15s. Lợi ích mỏng như vậy vì `gemini-3.5-flash` là model có suy luận nội bộ: nó nghĩ xong gần hết rồi mới bắt đầu phát token, nên phần lớn độ trễ nằm ở trước chunk đầu tiên và streaming không có gì để che. Điều này làm mình nhận ra streaming không phải phép màu chung cho mọi model — nó đáng giá khi độ trễ phân bố *dọc theo* quá trình sinh, tức câu trả lời dài và model phát token đều từ đầu, ví dụ viết một bài hướng dẫn nhiều đoạn nơi người dùng đọc kịp phần đầu trong lúc phần sau còn đang sinh; còn với model nghĩ trước rồi nói sau, hoặc với câu trả lời chỉ vài chục chữ, thì công sức xử lý chunk gần như không đổi lại được gì. Non-streaming hợp hơn ở mọi chỗ cần *toàn bộ* văn bản trước khi làm gì tiếp: khi phải parse JSON rồi mới validate, khi cần chạy kiểm duyệt nội dung trước lúc cho người dùng nhìn thấy (streaming thì chữ đã hiện rồi, rút lại rất khó coi), khi gọi từ job nền không có ai ngồi xem, và khi muốn code đơn giản vì xử lý chunk kéo theo hàng loạt tình huống phải lo như chunk cuối có `delta.content = None` mà bài lab đã cố tình gài vào mock.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Mình đo `retry_with_backoff` bằng một hàm luôn ném lỗi và in mốc thời gian: bốn lần gọi tại 0,000s / 0,101s / 0,302s / 0,703s, tức các khoảng chờ 0,101 — 0,201 — 0,401 giây đúng như công thức `base_delay * 2^attempt`, và `ValueError` gốc được ném lại nguyên vẹn chứ không bị nuốt. Lợi thế so với delay cố định là backoff tự thích ứng với mức độ nghiêm trọng: lỗi thoáng qua được cứu ngay ở lần chờ 0,1 giây nên ứng dụng không bị phạt oan, còn nếu server thật sự ngộp thì khoảng chờ giãn ra nhanh, tự giảm tải đúng lúc cần. Nếu hàng nghìn client cùng dùng delay cố định thì thành **thundering herd**: tất cả cùng hỏng gần như đồng thời, cùng chờ đúng 1 giây, rồi cùng đập vào server ở giây kế tiếp thành từng đợt sóng ngày càng đồng pha — đó là lý do hệ thống thật còn cộng thêm jitter ngẫu nhiên để phá vỡ sự đồng bộ đó. Có một chuyện xảy ra ngoài kịch bản mà mình muốn ghi lại: lúc chạy thí nghiệm Câu 4.1 mình dính 429 thật từ Gemini, và `retry_with_backoff` **không cứu nổi**, vì tổng thời gian chờ của nó chỉ 0,7 giây trong khi hạn mức free tier tính theo phút — bài học là backoff chỉ hợp với lỗi thoáng qua, còn gặp quota thì phải đọc `Retry-After` và chờ ở thang phút, hoặc đơn giản là đổi sang model nhỏ hơn như mình đã làm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt mình dùng là: "Bạn là trợ giảng của khóa AI dành cho người mới bắt đầu. Trả lời bằng tiếng Việt, tối đa 3 câu. Luôn kèm một ví dụ cụ thể. Nếu không chắc chắn, hãy nói rõ là không chắc thay vì đoán." Mình viết "tối đa 3 câu" thay vì "trả lời ngắn gọn" vì tính từ mơ hồ khiến độ dài trôi dần qua từng lượt, còn con số thì kiểm chứng được — và mình đã kiểm thật: chạy `run_assistant` với ba câu hỏi, cả ba phản hồi đều đúng 3 câu. Nhưng chính phép đo đó lại phơi ra chỗ mình nghĩ chưa tới: ba câu ấy dài 89, 82 và 93 từ, nghĩa là ràng buộc số câu **không hề làm câu trả lời ngắn đi**, model chỉ viết ba câu thật dài để lách. Nếu làm lại mình sẽ ghi "tối đa 60 từ" vì token mới là thứ đẻ ra chi phí, chứ không phải dấu chấm. Chỉ định "trả lời bằng tiếng Việt" thì cần thiết vì câu hỏi kỹ thuật hay lẫn thuật ngữ Anh (kiểu "streaming với retry khác gì nhau"), không ghim ngôn ngữ là model dễ trôi sang tiếng Anh giữa chừng. Cả phiên ba lượt tốn 316 token và $0,003063 theo `estimate_cost`.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là `history[-6:]`, và mình dựng hẳn một kịch bản năm lượt để xem nó hỏng ra sao: lượt 1 mình nói "Tên mình là Thực, mình đang học Python, nhớ tên mình nhé" và trợ lý đáp "Chào Thực, rất vui được làm quen với bạn"; sau ba lượt hỏi linh tinh, lượt 5 mình hỏi "Tên mình là gì?" và nhận được "Rất tiếc, mình chưa biết tên bạn vì bạn chưa giới thiệu với mình." Điều khiến mình thấy đây là lỗi nghiêm trọng không phải chuyện quên — quên thì hiểu được — mà là model **phủ nhận rằng mình từng được cho biết**, một câu sai hoàn toàn về mặt sự thật và làm người dùng mất lòng tin ngay lập tức. In `history` ra thì rõ nguyên nhân: 6 message còn lại bắt đầu từ lượt 3, tin nhắn chứa tên đã bị đẩy ra ngoài. Cắt cứng theo số message còn tệ ở chỗ nó không thật sự chặn được chi phí, vì ba lượt gần nhất mà dài thì input token vẫn phình, trong khi giới hạn thật của model là token chứ không phải số message. Cải thiện mình đề xuất là cắt theo ngân sách token kèm tóm tắt phần bị loại: thay `history[-6:]` bằng `trim_history(history, budget=2000)` duyệt từ mới về cũ và cộng dồn `count_tokens` cho tới khi chạm ngân sách; những message bị loại thì gom lại, gọi `call_openai_mini` với prompt "Tóm tắt hội thoại sau trong 2 câu, giữ lại các thông tin người dùng đã cung cấp" — dùng model nhỏ vì tóm tắt là việc dễ và rẻ hơn 16,67 lần như đã tính ở Câu 1.3 — rồi chèn kết quả ngay sau system prompt dưới dạng một message `system` ghi "Tóm tắt hội thoại trước đó: ...", để persona vẫn đứng đầu còn ngữ cảnh cũ tồn tại ở dạng nén. Đánh đổi là mỗi lần cắt tốn thêm một lời gọi API và bản tóm tắt vẫn có thể làm rơi chi tiết, nên chỉ nên tóm tắt khi thật sự vượt ngân sách chứ không phải sau mỗi lượt.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
